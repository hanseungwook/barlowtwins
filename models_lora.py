import math

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


##### LoRA MODULES #####

class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_scaling: float,
        lora_dropout: float,
        merge_weights: bool,
    ):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_scaling: lora scaling, note we don't use alpha here, instead directly set the scaling
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            merge_weights: whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
                fine-tuned model as a standalone one (without storing LoRA weights separately) plus it helps to reduce
                overhead during inference.
        """

        assert 0 <= r, f"LoRA rank must be positive, got {r}"
        assert (
            0.0 < lora_scaling <= 2.0
        ), f"LoRA scaling must be positive, got {lora_scaling}"

        self.r = r
        self.scaling = lora_scaling
        self.lora_dropout = lora_dropout
        # Optional dropout
        if self.lora_dropout > 0.0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Params4bit(bnb.nn.Params4bit):
    # as in bitsandbytes version 0.41.3, the original Params4bit has issue when moving model between CPU and GPU.
    # for example, when we try to move a quantized layer to CPU, and later move back to GPU, the weights would stay on CPU
    # https://github.com/TimDettmers/bitsandbytes/issues/902
    def cuda(self, device):
        if self.quant_state is not None:
            if self.data.device != device:
                self.data = self.data.to(device)
                self.quant_state.to(device)
            return self
        w = self.data.contiguous().half().cuda(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(
            w,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
        )
        self.data = w_4bit
        self.quant_state = quant_state
        return self

def transpose(weight: torch.Tensor, fan_in_fan_out: bool) -> torch.Tensor:
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class Linear4bit(bnb.nn.Linear4bit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = Params4bit(
            self.weight.data,
            requires_grad=False,
            compress_statistics=self.weight.compress_statistics,
            quant_type=self.weight.quant_type,
        )


class LoRALinear4bit(Linear4bit, LoRALayer):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        compress_statistics=True,
        quant_type="fp4",
        compute_dtype=None,
        device=None,
        r: int = 32,
        lora_scaling: float = 1.0,
        lora_dropout: float = 0.05,
        merge_weights: bool = True,
    ) -> None:
        Linear4bit.__init__(
            self,
            input_features=in_features,
            output_features=out_features,
            bias=bias,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            device=device,
        )

        LoRALayer.__init__(
            self,
            r=r,
            lora_scaling=lora_scaling,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Actual trainable parameters
        if r > 0:
            factory_kwargs = {"device": device, "dtype": compute_dtype}
            self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        # Don't reset the Linear4bit's weights here
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor):
        result = Linear4bit.forward(self, x)

        if self.r > 0:
            # dropout don't affect the model when in eval() mode
            result += (
                self.dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling

        return result


def get_layer(
    quantize: bool = False,
    lora: bool = False,
    r: int = 32,
    dropout: float = 0.05,
):
    if quantize and lora:
        return lambda *args, **kwargs: LoRALinear4bit(
            r=r, lora_dropout=dropout, *args, **kwargs
        )
    elif quantize and not lora:
        return Linear4bit
    elif not quantize and lora:
        return lambda *args, **kwargs: LoRALinear(
            r=r, lora_dropout=dropout, *args, **kwargs
        )
    else:
        return nn.Linear


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 32,
        lora_scaling: float = 1.0,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_scaling=lora_scaling,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
            self.lora_A = nn.Parameter(torch.empty((r, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, r), **factory_kwargs))
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        self.weight.data = transpose(self.weight.data, self.fan_in_fan_out)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_delta_weight(self) -> torch.Tensor:
        return transpose(self.lora_B @ self.lora_A, self.fan_in_fan_out) * self.scaling

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.get_delta_weight().type_as(self.weight)
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.get_delta_weight().type_as(self.weight)
                self.merged = True

    def forward(self, x: torch.Tensor):
        result = F.linear(
            x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

        if self.r > 0 and not self.merged:
            result += (
                self.dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling

        return result