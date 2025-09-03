import time
import torch
from itertools import chain
import torch.nn.functional as F
import glob
import os
from tqdm import tqdm
from data_processing import ik_util


def get_proprio_wrt_cam_onehandonly(batch, device, chirality, action_type):
    if action_type == "3d_joints_wrt_cam":
        proprio_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, 0:1, chirality, :, :]
        # note: aria cam wrt world is just the first sampled frame, which is the proprio
        proprio_wrt_world_homo = torch.cat([proprio_wrt_world, torch.ones((*proprio_wrt_world.shape[:-1], 1)).to(device)], axis=-1)

        # -> batch_size, 1, 21, 3
        proprio_wrt_currentcam = torch.einsum("bid, bhjd -> bhji", torch.linalg.inv(batch['aria_cam_wrt_world'][:, 0, :, :].to(torch.float32)), proprio_wrt_world_homo)[..., :3]
    else:
        # -> batch_size, horizon = 1, 1, 10
        proprio_wrt_world = batch['pseudogripper_wrt_world_10d_proprio'][:, 0:1, chirality, :]

        # -> batch_size, 10 -> batch_size, 1, 10
        proprio_wrt_currentcam = ik_util.apply_left_transform_to_10d(proprio_wrt_world.reshape(-1, 10), torch.linalg.inv(batch['aria_cam_wrt_world'][:, 0:1, :, :].to(torch.float32)).reshape(-1, 4, 4).to(proprio_wrt_world.device)).unsqueeze(1)
    return proprio_wrt_currentcam

def get_action_wrt_cam_onehandonly(batch, device, chirality, action_type):
    if action_type == "3d_joints_wrt_cam":
        action_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, 1:, chirality, :, :]
        action_wrt_world_homo = torch.cat([action_wrt_world, torch.ones((*action_wrt_world.shape[:-1], 1)).to(device)], axis=-1)
        # 0 is the current timestep
        action_wrt_currentcam = torch.einsum("bid, bhjd -> bhji", torch.linalg.inv(batch['aria_cam_wrt_world'][:, 0, :, :].to(torch.float32)), action_wrt_world_homo)[..., :3]
    else:
        action_wrt_world = batch['pseudogripper_wrt_world_10d_action'][:, 1:, chirality, :]
        action_horizon = action_wrt_world.shape[1]

        # -> batch_size, action_horizon, 10
        action_wrt_currentcam = ik_util.apply_left_transform_to_10d(action_wrt_world.reshape(-1, 10), torch.linalg.inv(batch['aria_cam_wrt_world'][:, 0:1, :, :].to(torch.float32)).expand(-1, action_horizon, -1, -1).reshape(-1, 4, 4).to(action_wrt_world.device)).reshape(-1, action_horizon, 10)

    return action_wrt_currentcam

def get_action_wrt_cam(batch, device, chirality=1):
    # -> batch size, 30, 2, 21, 3
    action_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, 1:, :, :, :]
    action_wrt_world_homo = torch.cat([action_wrt_world, torch.ones((*action_wrt_world.shape[:-1], 1)).to(device)], axis=-1)
    action_wrt_cam = torch.einsum("bid, bhcjd -> bhcji", torch.linalg.inv(batch['aria_cam_wrt_world'].to(torch.float32)), action_wrt_world_homo)

    return action_wrt_cam[..., :3]

def get_proprioaction_wrt_cam(batch, device, chilarity=1):
    # -> batch size, 30, 2, 21, 3
    action_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, :, :, :, :]
    action_wrt_world_homo = torch.cat([action_wrt_world, torch.ones((*action_wrt_world.shape[:-1], 1)).to(device)], axis=-1)
    action_wrt_cam = torch.einsum("bid, bhcjd -> bhcji", torch.linalg.inv(batch['aria_cam_wrt_world'].to(torch.float32)), action_wrt_world_homo)

    return action_wrt_cam[..., :3]

import torch
import torch.nn.functional as F

import numpy as np
from utils.se3_util import rot6d_to_matrix

def get_lerobot_homo_from_state_tensor(lerobot_state_tensor_wrt_world):
    """
    lerobot_state_tensor_wrt_world: num_samples, H, 10
        format: xyz, 6D, gripper
    return: num_samples, 4, 4
    """
    if isinstance(lerobot_state_tensor_wrt_world, np.ndarray):
        lerobot_state_tensor_wrt_world = torch.from_numpy(lerobot_state_tensor_wrt_world).to(torch.float32)
    
    ph_ = torch.eye(4).repeat(lerobot_state_tensor_wrt_world.shape[0], 1, 1).clone().to(lerobot_state_tensor_wrt_world.device)
    ph_[:, :3, 3] = lerobot_state_tensor_wrt_world[:, :3]
    ph_[:, :3, :3] = rot6d_to_matrix(lerobot_state_tensor_wrt_world[:, 3:9])
    return ph_

def get_lerobot_state_tensor_wrt_cam(lerobot_state_tensor_wrt_world, cam_wrt_world):
    """
    lerobot_state_tensor_wrt_world: num_samples, 10
    cam_wrt_world: 4x4
    return: num_samples, H, 10
    """
    raise NotImplementedError("This probably has a bug")
    return (torch.linalg.inv(cam_wrt_world.to(torch.float32).to(lerobot_state_tensor_wrt_world.device)) @ lerobot_state_tensor_wrt_world.mT).mT


def get_nproprio_from_batch(batch, dtype, batch_size, device, x_min, x_max, action_type):
    # proprio_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, 0:1, 1, :, :]
    # proprio_wrt_world_homo = torch.cat([proprio_wrt_world, torch.ones((*proprio_wrt_world.shape[:-1], 1)).to(device)], axis=-1)

    # # note: aria cam wrt world is just the first sampled frame, which is the proprio
    # proprio_wrt_cam = torch.einsum("bid, bhjd -> bhji", torch.linalg.inv(batch['aria_cam_wrt_world'].to(torch.float32)), proprio_wrt_world_homo)

    # # normalize...
    # # -> batch_size, 1, 21, 3
    # proprio_num_steps = proprio_wrt_cam.shape[1]

    # uproprio = proprio_wrt_cam[..., :3].reshape(batch_size, proprio_num_steps, 21, 3).to(dtype).to(device)

    proprio_wrt_cam = get_proprio_wrt_cam_onehandonly(batch, device, 1, action_type)
    proprio_num_steps = proprio_wrt_cam.shape[1]
    
    if action_type == "3d_joints_wrt_cam":
        uproprio = proprio_wrt_cam.view(batch_size, proprio_num_steps, 21, 3).to(device)
    else:
        uproprio = proprio_wrt_cam.view(batch_size, proprio_num_steps, 10).to(device)

    # print(f"Debug nproprio computation:")
    # print(f"  uproprio shape: {uproprio.shape}")
    # print(f"  uproprio[0]: {uproprio.flatten()[0]}")
    # print(f"  x_min shape: {x_min.shape}, x_max shape: {x_max.shape}")
    # print(f"  x_min[0]: {x_min.flatten()[0]}, x_max[0]: {x_max.flatten()[0]}")
    
    # # Step 1: Subtract x_min
    # step1 = uproprio - x_min.to(device)
    # print(f"  After subtracting x_min - [0]: {step1.flatten()[0]}")
    
    # # Step 2: Divide by range
    # range_val = x_max.to(device) - x_min.to(device)
    # step2 = step1 / range_val
    # print(f"  After dividing by range - [0]: {step2.flatten()[0]}")
    
    # # Step 3: Scale by 2
    # step3 = 2 * step2
    # print(f"  After scaling by 2 - [0]: {step3.flatten()[0]}")
    
    # # Step 4: Subtract 1
    # step4 = step3 - 1
    # print(f"  Final normalized (before reshape) - [0]: {step4.flatten()[0]}")
    nproprio = 2*(uproprio - x_min.to(device)) / (x_max.to(device) - x_min.to(device)) - 1

    if action_type == "3d_joints_wrt_cam":
        return nproprio.view(batch_size, proprio_num_steps, 63)
    else:
        return nproprio.view(batch_size, proprio_num_steps, 10)

def get_nactions_from_batch(batch, dtype, batch_size, device, x_min, x_max, action_type):
    # action_wrt_world = batch['ego_keypoint_3d_wrtworld_tensor'][:, 1:, 1, :, :]
    # action_wrt_world_homo = torch.cat([action_wrt_world, torch.ones((*action_wrt_world.shape[:-1], 1)).to(device)], axis=-1)
    # action_wrt_cam = torch.einsum("bid, bhjd -> bhji", torch.linalg.inv(batch['aria_cam_wrt_world'].to(torch.float32)), action_wrt_world_homo)

    action_wrt_cam = get_action_wrt_cam_onehandonly(batch, device, 1, action_type)
    action_num_steps = action_wrt_cam.shape[1]

    if action_type == "3d_joints_wrt_cam":
        uactions = action_wrt_cam.view(batch_size, action_num_steps, 21, 3).to(device)
    else:
        uactions = action_wrt_cam.view(batch_size, action_num_steps, 10).to(device)

    nactions = 2*(uactions - x_min.to(device)) / (x_max.to(device) - x_min.to(device)) - 1

    if action_type == "3d_joints_wrt_cam":
        return nactions.view(batch_size, action_num_steps, 63)
    else:
        return nactions.view(batch_size, action_num_steps, 10)

def normalize_tensor_3dplus(tensor_3d, x_min, x_max):
    # TODO: rename this, actually normalizes anything
    # this just does the above normalization outside the batch
    # actions: batch_size, horizon, 63 (?)
    batch_size = tensor_3d.shape[0]
    action_num_steps = tensor_3d.shape[1]
    action_dim_flat = tensor_3d.shape[2]

    data_shape = x_min.shape

    assert len(tensor_3d.shape) >= 3

    uactions = tensor_3d.reshape(batch_size, action_num_steps, *data_shape)
    nactions = 2*(uactions - x_min.to(uactions.device)) / (x_max.to(uactions.device) - x_min.to(uactions.device)) - 1
    return nactions.reshape(batch_size, action_num_steps, action_dim_flat)

def normalize_tensor_2d(tensor_2d, x_min, x_max):
    assert len(tensor_2d.shape) == 2
    # same as above but no horizon dimension
    batch_size = tensor_2d.shape[0]
    action_dim_flat = tensor_2d.shape[1]

    data_shape = x_min.shape
    uactions = tensor_2d.reshape(batch_size, *data_shape)
    nactions = 2*(uactions - x_min.to(uactions.device)) / (x_max.to(uactions.device) - x_min.to(uactions.device)) - 1
    return nactions.reshape(batch_size, action_dim_flat)

def unnormalize_action_proprio_vec(nactions_or_nproprios, x_min, x_max):
    assert len(nactions_or_nproprios.shape) >= 3
    if nactions_or_nproprios.dtype == torch.bfloat16:
        # this reduces reconstruction error from normalization by a factor of 10... Why?
        nactions_or_nproprios = nactions_or_nproprios.to(torch.float32)

    state_shape = x_min.shape
    # this just does the above normalization outside the batch
    # actions: batch_size, horizon, 63 (?)
    batch_size = nactions_or_nproprios.shape[0]
    action_num_steps = nactions_or_nproprios.shape[1]
    # uactions = 2*(nactions + 1)/2 * (x_max - x_min) + x_min
    # return uactions.reshape(batch_size, action_num_steps, 63)

    x_max = x_max.to(nactions_or_nproprios.device)
    x_min = x_min.to(nactions_or_nproprios.device)

    # # Debug prints for unnormalize_action_proprio_vec
    # print(f"Unnormalize debug - Input shape: {nactions_or_nproprios.shape}")
    # print(f"Unnormalize debug - Input first element: {nactions_or_nproprios.flatten()[0]}")
    
    # # Step 1: Add 1
    # step1 = nactions_or_nproprios.clone().reshape(batch_size, -1, 21, 3) + 1
    # print(f"  After adding 1 - first element: {step1.flatten()[0]}")
    
    # # Step 2: Divide by 2
    # step2 = step1 / 2
    # print(f"  After dividing by 2 - first element: {step2.flatten()[0]}")
    
    # # Step 3: Multiply by range
    # range_val = x_max - x_min
    # step3 = step2 * range_val
    # print(f"  After multiplying by range - first element: {step3.flatten()[0]}")
    
    # # Step 4: Add x_min
    # step4 = step3 + x_min
    # print(f"  Final unnormalized - first element: {step4.flatten()[0]}")
    uactions_or_uproprios = ((nactions_or_nproprios.clone().reshape(batch_size, -1, *state_shape) + 1)/2) * (x_max - x_min) + x_min
    return uactions_or_uproprios

def collate_batches_lerobot_dataset(batch_list):
    canonical_batch = batch_list[0]
    out_batch = dict()
    for key in canonical_batch.keys():
        if isinstance(canonical_batch[key], torch.Tensor):
            out_batch[key] = torch.concatenate([batch[key] for batch in batch_list], axis=0)
        elif isinstance(canonical_batch[key], list):
            out_batch[key] = list(chain.from_iterable([batch[key] for batch in batch_list]))
        else:
            raise NotImplementedError(f"Key {key} not implemented for lerobot dataset")
    return out_batch

def collate_batches_egoexo_dataset(batch_list, partition_size, expand_to_match_two_hands=True, index_with_bbox_presence=True, keys_to_collate=set(["rgb_cached_frames", "ego_keypoint_3d_wrtworld_tensor", "mano_params_tensor", "pseudogripper_wrt_world_10d_proprio", "pseudogripper_wrt_world_10d_action", "aria_cam_wrt_world"])):
    # input: list of samples
    # output: combine that list into a dictionary of samples

    # input batch has samples of tensors of form:
    # num_vids, num_frames_per_vid, 2, ...
    # we flatten the first three dimensions
    # some of the resulting samples will be unusable, due to missing bounding box
    # pass a boolean that we can use to index the valid samples

    """
    expand_to_match_two_hands: treat each hand (left/right) as a separate sample. Effectively doubles the samples, in the format: sample_1_left, sample_1_right, etc. I
        If we treat each hand label as a separate sample, some data types must be duplicated (i.e. a left and right hand label should have their own copies of the rgb image.)
    """
    # t0 = time.time()
    canonical_sample = {k: v for k, v in batch_list[0].items()}

    out_batch = dict()

    # # example: rgb_frames. 10, 1404, 1404, 3
    # key_to_use_dict maps from egoexo key to a new key name
    key_to_use_dict = dict(rgb_frames="image", rgb_cached_frames="cached_image", rgb_patches_tensor="image_patch", ego_keypoint_2d_patches_tensor="keypoints_2d_patch", ego_keypoint_2d_fullimg_tensor="keypoints_2d_fullimg",
    keypoint_present_tensor="visible", ego_bboxes_centers="center", ego_bboxes_sizes="bbox_size", ego_bboxes_tensor="bbox", ego_bboxes_confidences_tensor="confidence")

    # time the bbox presence computation
    # t_start = time.time()
    # -> batch_size, inflate_length, 2 -> batch_size * inflate_length * 2 (we want to flatten along the hand dimensiontoo)
    if "ego_bboxes_present_tensor" in canonical_sample.keys() and expand_to_match_two_hands:
        # bbox_presence = torch.concatenate([sample["ego_bboxes_present_tensor"] for sample in batch_list], axis=0).reshape(-1)
        bbox_presence = torch.concatenate([sample["ego_bboxes_present_tensor"].view(-1) for sample in batch_list], axis=0)
    else:
        # * partition_size?
        # when doing clip sampling, every subclip has been filtered
        bbox_presence = torch.ones(len(batch_list), dtype=torch.bool)
    
    # print(f"time taken to compute bbox presence: {time.time() - t_start}")

    out_batch['ego_bboxes_present_tensor'] = bbox_presence
    assert torch.sum(bbox_presence) > 0, "No bboxes found. Will be errors if we try to pass this batch further downstream"

    if expand_to_match_two_hands:
        # note: when this is used in dataloader, pytorch adds a dummy dimension to the tensors.
        # which is why we need to index into it with [0]
        for key in canonical_sample.keys():
            if isinstance(canonical_sample[key], torch.Tensor):
                key_to_use = key_to_use_dict.get(key, key)
                # we skip the first dimension, which is the number of videos
                # and we skip the second dimension, which is the number of hands
                if key == "rgb_frames" or key == "rgb_debug_frames":
                    # these are the only keys rn without a hand dimension
                    # we need to duplicate these 
                    # out_batch[key_to_use] = torch.concatenate([sample[key][0].unsqueeze(1).expand(-1, 2, -1, -1, -1) for sample in batch_list], axis=0).reshape(-1, *canonical_sample[key].shape[2:])
                    out_batch[key_to_use] = torch.concatenate([sample[key][0].unsqueeze(1).expand(-1, 2, -1, -1, -1).reshape(-1, *canonical_sample[key].shape[2:]) for sample in batch_list], axis=0)
                elif key == "sampled_frame_idxs_30fps":
                    # actually this also misses hand dimension
                    # sampled_frame_idxs_30fps: batch_size, inflate_length, 2 -> batch_size * inflate_length * 2
                    # out_batch[key_to_use] = torch.concatenate([sample[key].unsqueeze(-1).expand(-1, -1, 2) for sample in batch_list], axis=0).reshape(-1)
                    out_batch[key_to_use] = torch.concatenate([sample[key].unsqueeze(-1).expand(-1, -1, 2).reshape(-1) for sample in batch_list], axis=0)
                    num_flat_samples = out_batch['sampled_frame_idxs_30fps'].shape[0] // 2
                elif key == "aria_cam_wrt_world" or key == "aria_cam_intrinsics" or key == "exo_cam_wrt_world" or key == "exo_cam_intrinsics":
                    continue # don't need this for now
                    out_batch[key_to_use] = torch.concatenate([sample[key].reshape(-1, *canonical_sample[key].shape[2:]) for sample in batch_list], axis=0)
                else:
                    # ex. keypoint_2d_tensor: batch_size, inflate_length, 2, 21, 2 -> batch_size * inflate_length * 2, 21, 2
                    # out_batch[key_to_use] = torch.concatenate([sample[key] for sample in batch_list], axis=0).reshape(-1, *canonical_sample[key].shape[3:])
                    out_batch[key_to_use] = torch.concatenate([sample[key].reshape(-1, *canonical_sample[key].shape[3:]) for sample in batch_list], axis=0)

                if index_with_bbox_presence:
                    try:
                        out_batch[key_to_use] = out_batch[key_to_use][bbox_presence]
                    except:
                        print(f"Key {key_to_use} cannot be indexed by bbox_presence, skipping")
                        continue

            else:
                if key == "take_name" or key == "cam_name":
                    # double list to account for two hands. 
                    # Note: this will fail if we have multiple takes in a batch.
                    if index_with_bbox_presence:
                        out_batch[key] = [_ for sample_idx, _ in enumerate(list(chain.from_iterable([[_[0] for _ in sample[key]*2] for sample in batch_list]))) if bbox_presence[sample_idx]]
                    else:
                        out_batch[key] = list(chain.from_iterable([[_[0] for _ in sample[key]*2] for sample in batch_list]))
                else:
                    if not bool(batch_list[0][key]):
                        continue
                    else:
                        print(f"Key {key} logic not implemented, skipping")
                        continue
                        # if index_with_bbox_presence:
                        #     out_batch[key] = [sample[key] for sample_idx, sample in enumerate(batch_list) if bbox_presence[sample_idx]]
                        # else:
                        #     out_batch[key] = [sample[key] for sample in batch_list]
    else:
        """
        In this setting, for every RGB image, we just have one hand (for now) or a pair of hands (for future bimanual setting)
        """
        # note: in this mode we are not indexing away based on the bbox presence
        t327_start = time.perf_counter()
        for key in canonical_sample.keys() & keys_to_collate:
            t_start = time.perf_counter()
            if isinstance(canonical_sample[key], torch.Tensor):
                key_to_use = key_to_use_dict.get(key, key)
                out_batch[key_to_use] = torch.from_numpy(np.concatenate([sample[key] for sample in batch_list], axis=0).reshape(-1, *canonical_sample[key].shape[2:]))
            elif key == "take_name" or key == "cam_name":
                # flattens a nested list into a single list
                out_batch[key] = list(chain.from_iterable([[_[0] for _ in sample[key]] for sample in batch_list]))
            elif key == "text_labels":
                out_batch[key] = list(chain.from_iterable([[_ for _ in sample[key]] for sample in batch_list]))
            else:
                # TODO: check this is right
                out_batch[key] = [sample[key] for sample in batch_list]
            # print(f"time taken to collate {key}: {time.perf_counter() - t_start}")

    # print(f"ln352 time taken to collate fn: {time.perf_counter() - t327_start}")
    if expand_to_match_two_hands:
        # -> batch_size, inflate_length, 2

        tmp = torch.zeros(num_flat_samples, 2)
        tmp[:, 1] = 1
        out_batch["is_right"] = tmp.reshape(-1).bool()
        
        if index_with_bbox_presence:
            out_batch["is_right"] = out_batch["is_right"][bbox_presence]
    # else:
    #     out_batch["is_right"] = torch.zeros(num_flat_samples, 2)


    return out_batch


# ---------------------------------------------------------------------------
# Data augmentation utilities
# ---------------------------------------------------------------------------
# from torchvision.transforms import ColorJitter

import torchvision.transforms.v2 as T   # ≥ torchvision 0.17

_COLOR_JITTER = T.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
)

_RANDOM_TRANSLATION = T.RandomResizedCrop(size=(224, 224), scale=(0.95, 1.0), ratio=(.95, 1.0), interpolation=T.InterpolationMode.BILINEAR, antialias=True)

def colorjitter_augmentation(patches: torch.Tensor) -> torch.Tensor:
    """
    Same contract as before, but ~10-100× faster by:
      • getting rid of the Python loop  
      • working on a float tensor once  
      • letting torchvision handle per-image randomness in batch
    """
    assert patches.dtype == torch.uint8 and patches.shape[-1] == 3
    orig_shape = patches.shape                                    # (..., H, W, 3)

    # (N, 3, H, W) float32 in [0, 1]
    x = patches.view(-1, *orig_shape[-3:]).permute(0, 3, 1, 2).float() / 255.

    # optional: x = x.cuda(non_blocking=True)  # GPU colour jitter with v2 or Kornia
    x = _COLOR_JITTER(x)                       # batched, independent jitter per image

    out = (x.mul(255).clamp(0, 255).byte()     # back to uint8, channel-last, orig shape
                .permute(0, 2, 3, 1)
                .view(orig_shape))
    return out


def cycle_shorter(longer, shorter):
    """
    For a longer and shorter iterator, yield the index, the longer element, and the shorter element.
    If the shorter iterator is exhausted, it will be reset.
    """
    # shorter_iter = iter(shorter)

    idx = 0
    # for long_el in tqdm.tqdm(iter(longer)):
    # for long_el in iter(longer):
    for long_el in longer:
        try:
            short_el = next(shorter)
        except StopIteration:
            # shorter_iter = iter(shorter)
            # short_el = next(shorter)
            short_el = next(shorter)

        yield idx, long_el, short_el
        idx += 1

def sample_fm_time(bsz: int, flow_sampling: str, flow_beta_dist: torch.distributions.Distribution, flow_t_max: float) -> torch.FloatTensor:
    if flow_sampling == "uniform":  # uniform between 0 and 1
        """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
        eps = 1e-5
        t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
    elif flow_sampling == "beta":  # from pi0 paper
        z = flow_beta_dist.sample((bsz,))
        t = flow_t_max * (1 - z)  # flip and shift
    return t
def convert_to_bfloat16(item, dtype):
    if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
        return item.to(dtype)
    elif isinstance(item, dict):
        return {k: convert_to_bfloat16(v, dtype) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_to_bfloat16(v, dtype) for v in item]
    else:
        return item
    

def save_checkpoint(model, optimizer, epoch, global_step, loss, checkpoint_dir, max_checkpoints=5):
    """Save checkpoint and manage checkpoint limit"""
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step:06d}.pt"
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        # 'config': OmegaConf.to_container(cfg, resolve=False)
    }, checkpoint_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Clean up old checkpoints
    checkpoint_files = sorted(glob.glob(str(checkpoint_dir / "checkpoint_step_*.pt")))
    if len(checkpoint_files) > max_checkpoints:
        files_to_delete = checkpoint_files[:-max_checkpoints]
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"Deleted old checkpoint: {file_path}")
    
    return checkpoint_path

def save_latest_checkpoint(model, optimizer, epoch, global_step, loss, checkpoint_dir, filename="latest.pt"):
    """Save latest checkpoint (overwrites previous latest)"""
    latest_path = checkpoint_dir / filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        # 'config': OmegaConf.to_container(cfg, resolve=False)
    }, latest_path)
    
    return latest_path

def save_best_model(model, optimizer, epoch, global_step, train_loss, val_loss, checkpoint_dir):
    """Save best model checkpoint (doesn't count toward checkpoint limit)"""
    best_model_path = checkpoint_dir / f"step_{global_step:06d}_best_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, best_model_path)
    
    print(f"Saved best model: {best_model_path}")
    return best_model_path

def validate_model(model, val_dataloader, processor, tokenizer, source_dataset, fast_tokenizer, fast_idx_to_gemma_idx, 
                    flow_sampling, flow_beta_dist, flow_t_max, dtype, use_pi_5_stopgrad, args):
    """Run validation on the validation dataset"""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        valid_batch_counter = 0
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc='Validation')):                
            if valid_batch_counter % args.accumulation_steps == 0:
                batch_queue = []            
                valid_batch_counter = 0

            if (valid_batch_counter + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(val_dataloader): # TODO: double check this... we might miss a batch if the last batches are empty and not hit this
                batch_queue.append(batch)
                batch = collate_batches_egoexo_dataset(batch_queue, None, expand_to_match_two_hands=False, index_with_bbox_presence=False)
                valid_batch_counter += 1
            else:
                batch_queue.append(batch)
                valid_batch_counter += 1
                continue

            batch_size = batch["cached_image"].shape[0]
            text_arr = ["test" for _ in range(batch_size)]

            model_inputs = processor(text=text_arr, images=batch["cached_image"].permute(0, 3, 1, 2))
            batch = convert_to_bfloat16(batch, dtype)
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            nproprio = get_nproprio_from_batch(batch, dtype, batch_size, next(model.parameters()).device, 
                                                source_dataset.x_min_wrt_cam_running, source_dataset.x_max_wrt_cam_running).to(dtype)
            nactions = get_nactions_from_batch(batch, dtype, batch_size, next(model.parameters()).device, 
                                                source_dataset.x_min_wrt_cam_running, source_dataset.x_max_wrt_cam_running).to(dtype)

            if use_pi_5_stopgrad:
                nactions_tokens_wrt_fast_indices = fast_tokenizer(nactions.float().data.cpu().numpy())
                nactions_tokens_wrt_gemma_indices = [[fast_idx_to_gemma_idx[fast_idx] for fast_idx in sentence] 
                                                    for sentence in nactions_tokens_wrt_fast_indices]

                dtm_ph = torch.zeros_like(model_inputs['attention_mask'])
                model_inputs['discrete_token_mask'] = dtm_ph
                for sentence_idx, sentence in enumerate(model_inputs["input_ids"]):
                    original_text_end = torch.nonzero(model_inputs['input_ids'][sentence_idx] == 108).item()
                    current_discrete_tokens_sentence_len = len(nactions_tokens_wrt_gemma_indices[sentence_idx])
                    model_inputs['input_ids'][sentence_idx][original_text_end+1:original_text_end+1+current_discrete_tokens_sentence_len+1] = torch.Tensor(nactions_tokens_wrt_gemma_indices[sentence_idx] + [108]).to(next(model.parameters()).device).long()
                    model_inputs['discrete_token_mask'][sentence_idx][original_text_end+1:original_text_end+1+current_discrete_tokens_sentence_len+1] = 1

            causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = model.build_causal_mask_and_position_ids(
                model_inputs['attention_mask'], dtype=dtype, 
                discrete_action_mask=model_inputs['discrete_token_mask'] if use_pi_5_stopgrad else None)

            image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

            inputs = {
                "input_ids": model_inputs["input_ids"],
                "pixel_values": model_inputs["pixel_values"].to(dtype),
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "action_position_ids": action_position_ids,
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
                "causal_mask": causal_mask,
                "t": sample_fm_time(len(text_arr), flow_sampling, flow_beta_dist, flow_t_max).to(dtype)
            }
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            batch.update(inputs)
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


            # loss, extra_loss_dict = model(egoexo_batch['input_ids'],
            #                     egoexo_batch['pixel_values'],
            #                     egoexo_batch['causal_mask'],
            #                     egoexo_batch['vlm_position_ids'],
            #                     egoexo_batch['proprio_position_ids'],
            #                     egoexo_batch['action_position_ids'],
            #                     [egoexo_nproprio, robot_nproprio],
            #                     [egoexo_nactions, robot_nactions],
            #                     egoexo_batch['t'],
            #                     use_pi_5_stopgrad=args.use_pi_5_stopgrad,
            #                     nactions_tokens_wrt_gemma_indices=nactions_tokens_wrt_gemma_indices if args.use_pi_5_stopgrad else None,
            #                     discrete_action_mask=model_inputs['discrete_token_mask'] if args.use_pi_5_stopgrad else None,
            #                     embodiment_loss_weights=cfg.embodiment_loss_weights)
            
            loss, extra_loss_dict = model(batch['input_ids'], 
                                          batch['pixel_values'], 
                                          batch['causal_mask'],
                                          batch['vlm_position_ids'], 
                                          batch['proprio_position_ids'], 
                                          batch['action_position_ids'],
                                          [nproprio, None], 
                                          [nactions, None], 
                                          batch['t'], 
                                          use_pi_5_stopgrad=use_pi_5_stopgrad,
                                          nactions_tokens_wrt_gemma_indices=nactions_tokens_wrt_gemma_indices if use_pi_5_stopgrad else None,
                                          discrete_action_mask=model_inputs['discrete_token_mask'] if use_pi_5_stopgrad else None)

            total_val_loss += loss.item()
            num_val_batches += 1

    model.train()  # Set back to training mode
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
    print(f"Validation completed: {num_val_batches} batches, avg loss = {avg_val_loss:.6f}")
    return avg_val_loss


def compute_lerobot_normalization(lerobot_dataset, calib):
    print("Building lerobot normalizer...")
    lerobot_dataset.hf_dataset.set_format(type="numpy", columns=["action", "observation.state"])
    # -> num_samples, action_dim
    all_actions = torch.from_numpy(np.stack(lerobot_dataset.hf_dataset["action"])).to(torch.float32)
    all_proprio = torch.from_numpy(np.stack(lerobot_dataset.hf_dataset["observation.state"])).to(torch.float32)

    if calib is not None:
        # if args.transform_lerobot_actions_to_camera_frame:
        # transform actions into camera frame
        # 1. cascade the actions into clips. Actually we don't need to do this, since the camera frame is fixed, all actions will just be wrt the fixed camrea.
        # 2. convert the clips into camera frame. Since the camera doesn't move, this just corresponds to applying a single transform to all the clips
        # 3. take the min and max of the clips
        print("Transforming lerobot actions to camera frame")
        # all_actions_homo = get_lerobot_homo_from_state_tensor(all_actions)
        # all_proprio_homo = get_lerobot_homo_from_state_tensor(all_proprio)

        # # -> num_samples, 4, 4
        # all_actions_wrt_cam = get_lerobot_state_tensor_wrt_cam(all_actions_homo, torch.from_numpy(dict(calib)['T_cam_wrt_robot']))
        # all_proprio_wrt_cam = get_lerobot_state_tensor_wrt_cam(all_proprio_homo, torch.from_numpy(dict(calib)['T_cam_wrt_robot']))
        all_actions_wrt_cam_10d = ik_util.apply_left_transform_to_10d(all_actions.reshape(-1, 10).float(), torch.linalg.inv(torch.from_numpy(dict(calib)['T_cam_wrt_robot']).float().unsqueeze(0)))
        all_proprio_wrt_cam_10d = ik_util.apply_left_transform_to_10d(all_proprio.reshape(-1, 10).float(), torch.linalg.inv(torch.from_numpy(dict(calib)['T_cam_wrt_robot']).float().unsqueeze(0)))

        # reassemble the 10d state
        # all_actions_wrt_cam_10d = torch.concatenate([all_actions_wrt_cam[..., :3, 3], all_actions_wrt_cam[..., :3, 0], all_actions_wrt_cam[..., :3, 1],  all_actions[..., 9:10]], axis=-1)
        # all_proprio_wrt_cam_10d = torch.concatenate([all_proprio_wrt_cam[..., :3, 3], all_proprio_wrt_cam[..., :3, 0], all_proprio_wrt_cam[..., :3, 1],  all_proprio[..., 9:10]], axis=-1)

        # don't forget to add gripper action back!!!
        lerobot_action_min, lerobot_action_max = np.percentile(all_actions_wrt_cam_10d, 0, axis=0), np.percentile(all_actions_wrt_cam_10d, 100, axis=0)
        lerobot_proprio_min, lerobot_proprio_max = np.percentile(all_proprio_wrt_cam_10d, 0, axis=0), np.percentile(all_proprio_wrt_cam_10d, 100, axis=0)
    else:
        lerobot_action_min, lerobot_action_max = np.percentile(all_actions, 0, axis=0), np.percentile(all_actions, 100, axis=0)
        lerobot_proprio_min, lerobot_proprio_max = np.percentile(all_proprio, 0, axis=0), np.percentile(all_proprio, 100, axis=0)


    lerobot_action_min = torch.from_numpy(lerobot_action_min)
    lerobot_action_max = torch.from_numpy(lerobot_action_max)
    lerobot_proprio_min = torch.from_numpy(lerobot_proprio_min)
    lerobot_proprio_max = torch.from_numpy(lerobot_proprio_max)

    print("Manually setting the gripper min/max")
    lerobot_proprio_max[-1] = 70.0
    lerobot_proprio_min[-1] = 0.0
    lerobot_action_max[-1] = 1.0
    lerobot_action_min[-1] = -1.0

    print("Done building lerobot normalizer")
    # return lerobot_action_min, lerobot_action_max, lerobot_proprio_min, lerobot_proprio_max
    return dict(lerobot_action_min=lerobot_action_min, 
    lerobot_action_max=lerobot_action_max, 
    lerobot_proprio_min=lerobot_proprio_min,
    lerobot_proprio_max=lerobot_proprio_max,
    all_actions_wrt_cam_10d=all_actions_wrt_cam_10d if calib is not None else None,
    all_proprio_wrt_cam_10d=all_proprio_wrt_cam_10d if calib is not None else None)


# def build_param_groups(model, vision_encoder_lr, remaining_lr):
#     """
#     Build param groups where the vision encoder is trained at different learning rate
#     """
#     for module, name in model.named_modules():

def multiples_up_to(nums, N):
    return [x * k for x in nums for k in range(1, N + 1)]
