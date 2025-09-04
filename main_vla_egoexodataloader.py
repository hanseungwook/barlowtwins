# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import datetime

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader


from models import *
# from ego_utils.egodata import EgoExoDatasetFromGTJson
from ego_utils.egoexo_dataset_from_gt_json import EgoExoDatasetFromGTJson
from omegaconf import DictConfig
from omegaconf import OmegaConf, open_dict
import train_util
import iter_util

from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

# model args
parser.add_argument('--language-vocab-size', default=10000, type=int, metavar='N',
                    help='language vocabulary size')
parser.add_argument('--action-vocab-size', default=10000, type=int, metavar='N',
                    help='action vocabulary size')
parser.add_argument('--language-hidden-dim', default=1024, type=int, metavar='N',
                    help='language hidden dimension')
parser.add_argument('--action-hidden-dim', default=1024, type=int, metavar='N',
                    help='action hidden dimension')
parser.add_argument('--layer-norm-eps', default=1e-12, type=float, metavar='N',
                    help='layer norm epsilon')

# ego data args
parser.add_argument('--num-subclips-per-video', default=2, type=int, metavar='N',
                    help='number of subclips per video')
parser.add_argument('--debug-max-takes', default=1, type=int, metavar='N',
                    help='number of takes to debug')
parser.add_argument('--three-d-keypoints-torch-root', default=None, type=str, metavar='N',
                    help='path to three d keypoints torch root')
parser.add_argument('--rectified-ego-focal-length', default=822, type=int, metavar='N',
                    help='rectified ego focal length')


parser.add_argument("config", type=Path, metavar='CONFIG',
                    help='path to config file')
parser.add_argument('--accumulation_steps', default=8, type=int)
parser.add_argument('--num_workers', default=8, type=int)
                    


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    # TODO: fix this
    # torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    main_worker(0, args)


def main_worker(gpu, args):
    """
    Start dataloading logic
    """
    # accumulation steps is the true number of videos per batch
    # batch_size: accumulation steps * num_subclips_per_video
    if args.accumulation_steps is None:
        args.accumulation_steps = args.num_workers

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now",
        lambda fmt="%Y-%m-%d_%H-%M-%S": datetime.now().strftime(fmt),
    )
    cfg = DictConfig(OmegaConf.load(args.config))

    # egoexo is human only data
    # robot is robot only data
    # both is human and robot data

    iter_type = cfg.training.iter_type
   
    if iter_type in ["egoexo", "both"]:
        egoexo_source_dataset = EgoExoDatasetFromGTJson(
            release_dir="/mnt/nfs_csail/hamer_diffusion_policy_project/datasets/egoexo/",
            annotation_root="/mnt/nfs_csail/models/rli14/data/egoexo/annotations/ego_pose/train/hand/annotation",
            bbox_npy_root="/mnt/nfs_csail/hamer_diffusion_policy_project/datasets/egoexo_generated/0630_egg_and_tomato_full",
            num_subclips_per_video=cfg.egoexo_dataset.num_subclips_per_video, # clips per batch / video
            render_debug_frames=False,
            debug_max_takes=args.debug_max_takes,
            return_full_images=False,
            allow_takes_with_annotations_only=False,
            allowed_take_names=cfg.egoexo_dataset.allowed_take_names,
            frames_type_to_use="valid_3d_kp_clips",
            hand_detector=None,
            partition_size=None, 
            early_return_rectified_frames=False,
            allowed_parent_tasks=["Cooking"],
            allowed_task_names=["Cooking Tomato & Eggs"],
            cam_type_to_use="ego",
            right_hand_bbox_scaler=None,
            left_hand_bbox_scaler=None,
            # colorjitter_augmentation=False,
            three_d_keypoints_torch_root=cfg.egoexo_dataset.three_d_keypoints_torch_root,
            cached_rgb_dir="/mnt/nfs_csail/hamer_diffusion_policy_project/datasets/egoexo_generated/0705_egg_and_tomato_full_rectified_export/",
            filter_clip_proprio_norm=.05,
            joint_wrt_cam_cache_dir=args.checkpoint_dir,
            rectified_ego_focal_length=args.rectified_ego_focal_length,
            load_cam_data=True,
            length_scale_factor=cfg.egoexo_dataset.length_scale_factor,
            text_annotation_json_path=cfg.egoexo_dataset.text_annotation_json_path,
            max_period_to_associate_timestamp=5,
            sample_clips_with_text_annotations_only=cfg.egoexo_dataset.sample_clips_with_text_annotations_only,
            load_mano_params_from_cache=cfg.egoexo_dataset.load_mano_params_from_cache
        )
    else:
        egoexo_source_dataset = None

    """ Start robot data loading logic"""
    # TODO: we can define train/val by the episodes argument into LeRobotDataset which filters it out
    # lerobot_train_dataset = LeRobotDataset(repo_id=cfg.training.lerobot_repo_id, delta_timestamps = {"observation.image.ego_global": [0], # get the current timestep image
    #                                                                                                 "observation.state": [0], # get the current timestep state
    #                                                                                                 "action": (np.arange(30) * 1/30).tolist() # get one second horizon of actions
    #                                                                                                 },
    #                                                                                                 force_cache_sync=True)

    # if hasattr(cfg.training, "transform_lerobot_actions_to_camera_frame") and cfg.training.transform_lerobot_actions_to_camera_frame:
    #     calib_path = cfg.training.cam_wrt_world_path
    #     print("Manually loading one extrinsic from: ", calib_path)
    #     calib = np.load(calib_path)

    #     T_cam_wrt_robot = torch.from_numpy(dict(calib)['T_cam_wrt_robot']).float()
    # else:
    #     calib = None
    #     T_cam_wrt_robot = None
    # if args.resume_checkpoint:
    #     lerobot_normalization_params_path = args.resume_checkpoint.replace(os.path.basename(args.resume_checkpoint), f"{cfg.training.normalization_type}_normalization_params.pt")
    #     lerobot_normalization_dict = torch.load(lerobot_normalization_params_path)
    # else:
        # lerobot_normalization_dict = compute_lerobot_normalization(lerobot_train_dataset, calib)
    # lerobot_action_min, lerobot_action_max, lerobot_proprio_min, lerobot_proprio_max = lerobot_normalization_dict["lerobot_action_min"], lerobot_normalization_dict["lerobot_action_max"], lerobot_normalization_dict["lerobot_proprio_min"], lerobot_normalization_dict["lerobot_proprio_max"]

    # at this point, normalization parameters should be calculated
    # TODO: save them to the checkpoint dir

    # we need to reload the dataset because it fucks up somehow when using the set format
    # lerobot_train_dataset = LeRobotDataset(repo_id=cfg.training.lerobot_repo_id, delta_timestamps = {"observation.image.ego_global": [0], # get the current timestep image
    #                                                                                             "observation.state": [0], # get the current timestep state
    #                                                                                             "action": (np.arange(30) * 1/30).tolist() # get one second horizon of actions
    #                                                                                             })
    #                                     #root=cfg.training.lerobot_train_root)

    # if cfg.training.get("load_lerobot_into_memory", True):
    #     # load everything into numpy arrays in the main process instead of decoding RGB
    #     lerobot_train_dataset.hf_dataset.set_format(type="numpy", columns=["action", "observation.state", "observation.image.ego_global"])
    #     lerobot_train_dataset_start_time = time.perf_counter()

    #     blob = lerobot_train_dataset.hf_dataset[:]

    #     # time this
    #     # lerobot_observation_image_ego_global = np.stack(lerobot_train_dataset.hf_dataset['observation.image.ego_global'])
    #     # lerobot_action = np.stack(lerobot_train_dataset.hf_dataset['action'])
    #     # lerobot_observation_state = np.stack(lerobot_train_dataset.hf_dataset['observation.state'])
    #     lerobot_observation_image_ego_global = blob['observation.image.ego_global']
    #     lerobot_action = blob['action']
    #     lerobot_observation_state = blob['observation.state']
    #     lerobot_train_dataset_end_time = time.perf_counter()
    #     print(f"Time taken to load lerobot into memory: {lerobot_train_dataset_end_time - lerobot_train_dataset_start_time} seconds")

    #     # match the egoexo batch size
    #     # accumulation steps is the batch size for egoexo
    #     lerobot_batch_size = args.accumulation_steps * args.num_subclips_per_video
    """ End robot data loading logic"""

    """ Split the source dataset into train and val indices"""
    if iter_type in ["egoexo", "both"]:
        # selected take names is the filtered cooking videos with clean hand labels
        all_indices = torch.arange(len(egoexo_source_dataset.selected_take_names)).int()

        # if args.resume_checkpoint and args.keep_data_split:
        #     val_take_names = metadata_dict["val_take_names"]
        #     egoexo_train_take_names = metadata_dict["train_take_names"]
        #     egoexo_val_indices = torch.tensor([egoexo_source_dataset.take_name_to_take_idx[take_name] for take_name in val_take_names])
        #     egoexo_train_indices = torch.tensor([egoexo_source_dataset.take_name_to_take_idx[take_name] for take_name in egoexo_train_take_names])

        #     assert args.debug_max_takes == len(egoexo_train_take_names)+ len(val_take_names), "Debug max takes must be the sum of train and val take names"
        # else:
        # sample validation indices uniformly (but with seed) as 5% of the datset
        num_val = max(1, int(len(egoexo_source_dataset.selected_take_names)*.05))
        assert num_val >= 1, "No >= 1 validation indices"
        egoexo_val_indices = torch.randperm(len(egoexo_source_dataset.selected_take_names))[:num_val]

        assert len(egoexo_val_indices) > 0, "No validation indices"

        val_take_names = [egoexo_source_dataset.take_idx_to_take_name[idx] for idx in egoexo_val_indices.tolist()]


        # train_indices is the complement of val_indices
        assert len(all_indices) > 0
        if len(all_indices) == 1:
            egoexo_train_indices = all_indices
            print("Only one take, using it for train and val")
        else:
            egoexo_train_indices = all_indices[~torch.isin(all_indices, egoexo_val_indices).bool()]
        
        egoexo_train_take_names = [egoexo_source_dataset.take_idx_to_take_name[idx] for idx in egoexo_train_indices.tolist()]
    

        # the indices might change as more videos are loaded in
        # wandb.config.update({"egoexo_val_indices": egoexo_val_indices.tolist(),
        #                     "egoexo_val_take_names": val_take_names}, allow_val_change=True)
        # wandb.config.update({"egoexo_train_indices": egoexo_train_indices.tolist(),
        #                      "egoexo_train_take_names": egoexo_train_take_names}, allow_val_change=True)


        # generate the training and validaton subsets from the source dataset using the sampled indices
        egoexo_train_dataset = Subset(egoexo_source_dataset, indices=train_util.multiples_up_to(egoexo_train_indices.tolist(), cfg.egoexo_dataset.length_scale_factor))
        egoexo_val_dataset = Subset(egoexo_source_dataset, indices=egoexo_val_indices.tolist())

        # the action type
        # 3d joints wrt cam is the 3d position vectors of the hand joints in the camera frame
        assert cfg.training.egoexo_action_type in ["3d_joints_wrt_cam", "pseudogripper_10d"], f"Invalid egoexo action type: {cfg.training.egoexo_action_type}"

        # the normalization type
        # using only egoexo data, only robot data, or the union of both data
        assert cfg.training.normalization_type in ["egoexo", "robot", "both"], f"Invalid normalization type: {cfg.training.normalization_type}"

        # calculate normalization parameters
        if cfg.training.egoexo_action_type == "3d_joints_wrt_cam":
            # TODO: this is a hack for human data b/c they are equal, but not true in robot data
            egoexo_action_wrt_cam_collected = egoexo_source_dataset.set_joints_wrt_cam_normalization_from_take_names(egoexo_train_take_names)
            egoexo_proprio_wrt_cam_collected = egoexo_action_wrt_cam_collected
        elif cfg.training.egoexo_action_type == "pseudogripper_10d":
            egoexo_proprio_wrt_cam_collected, egoexo_action_wrt_cam_collected = egoexo_source_dataset.set_pseudogripper_10d_normalization_from_take_names(egoexo_train_take_names)
        else:
            raise NotImplementedError

    """End split source dataset"""

    """
    Start normalization logic
    """
    if cfg.training.normalization_type in ["robot", "both"]:
        if cfg.training.normalization_type == "both":
            if args.resume_checkpoint:
                print("Resuming from checkpoint, skipping normalization calculation")
                lerobot_action_min = lerobot_normalization_dict["lerobot_action_min"]
                lerobot_action_max = lerobot_normalization_dict["lerobot_action_max"]
                lerobot_proprio_min = lerobot_normalization_dict["lerobot_proprio_min"]
                lerobot_proprio_max = lerobot_normalization_dict["lerobot_proprio_max"]
            else:
                robot_actions_wrt_cam_10d = lerobot_normalization_dict["all_actions_wrt_cam_10d"]
                robot_proprio_wrt_cam_10d = lerobot_normalization_dict["all_proprio_wrt_cam_10d"]

                print("Combining robot and human parameters")
                # update both lerobot and egoexo normalization parameters
                # total_actions = torch.cat([robot_actions_wrt_cam_10d, egoexo_action_wrt_cam_collected], axis=0)
                # total_proprio = torch.cat([robot_proprio_wrt_cam_10d, egoexo_proprio_wrt_cam_collected], axis=0)

                # remove outliers in the hand data, but do not screw up the robot data
                lerobot_action_min = torch.min(torch.cat([robot_actions_wrt_cam_10d, torch.quantile(egoexo_action_wrt_cam_collected, q=0.01, dim=0)[None, :]], dim=0), dim=0)[0]
                lerobot_action_max = torch.max(torch.cat([robot_actions_wrt_cam_10d, torch.quantile(egoexo_action_wrt_cam_collected, q=.99, dim=0)[None, :]], dim=0), dim=0)[0]
                lerobot_proprio_min = torch.min(torch.cat([robot_proprio_wrt_cam_10d, torch.quantile(egoexo_proprio_wrt_cam_collected, q=0.01, dim=0)[None, :]], dim=0), dim=0)[0]
                lerobot_proprio_max = torch.max(torch.cat([robot_proprio_wrt_cam_10d, torch.quantile(egoexo_proprio_wrt_cam_collected, q=.99, dim=0)[None, :]], dim=0), dim=0)[0]

                # we have to manually set the normalization of the gripper dimension or it will get erased by the q
                lerobot_action_min[-1] = -1
                lerobot_action_max[-1] = +1
                lerobot_proprio_min[-1] = 0
                lerobot_proprio_max[-1] = +70 # 70mm  max 

                assert torch.all(lerobot_action_max - lerobot_action_min != 0), "Action min and max are the same"
                assert torch.all(lerobot_proprio_max - lerobot_proprio_min != 0), "Proprio min and max are the same"

                # save to checkpoint dir
                torch.save({
                    "lerobot_action_min": lerobot_action_min,
                    "lerobot_action_max": lerobot_action_max,
                    "lerobot_proprio_min": lerobot_proprio_min,
                    "lerobot_proprio_max": lerobot_proprio_max,
                    "egoexo_action_wrt_cam_collected": egoexo_action_wrt_cam_collected,
                    "egoexo_proprio_wrt_cam_collected": egoexo_proprio_wrt_cam_collected
                }, os.path.join(checkpoint_dir, "both_normalization_params.pt"))


                # save to wandb
                wandb.config.update({
                    "lerobot_action_min": lerobot_action_min.tolist(),
                    "lerobot_action_max": lerobot_action_max.tolist(),
                    "lerobot_proprio_min": lerobot_proprio_min.tolist(),
                    "lerobot_proprio_max": lerobot_proprio_max.tolist()
                }, allow_val_change=False)

            # use the action min max for egoexo too, since we have "both"
            egoexo_source_dataset.action_min_wrt_cam_running = lerobot_action_min
            egoexo_source_dataset.action_max_wrt_cam_running = lerobot_action_max
            egoexo_source_dataset.proprio_min_wrt_cam_running = lerobot_proprio_min
            egoexo_source_dataset.proprio_max_wrt_cam_running = lerobot_proprio_max
        elif cfg.training.normalization_type == "robot":
            # at least serialize the robot normalization parameters
            torch.save({
                "lerobot_action_min": lerobot_action_min,
                "lerobot_action_max": lerobot_action_max,
                "lerobot_proprio_min": lerobot_proprio_min,
                "lerobot_proprio_max": lerobot_proprio_max
            }, os.path.join(checkpoint_dir, "lerobot_normalization_params.pt"))
        elif cfg.training.normalization_type == "egoexo":
            raise NotImplementedError
            # at least serialize the egoexo normalization parameters
            torch.save({
                "egoexo_action_min": egoexo_action_wrt_cam_collected.min(dim=0)[0],
                "egoexo_action_max": egoexo_action_wrt_cam_collected.max(dim=0)[0],
                "egoexo_proprio_min": egoexo_proprio_wrt_cam_collected.min(dim=0)[0],
                "egoexo_proprio_max": egoexo_proprio_wrt_cam_collected.max(dim=0)[0]
            }, os.path.join(checkpoint_dir, "egoexo_normalization_params.pt"))

    """
    End normalization logic
    """

    """
    Start worker distribution logic
        If either human or robot, assign all CPU workers to that dataloader.
        If both, and robot is in video mode, assign half. Else if robot is in memory, all all workers to CPU.
    """
    if iter_type == "robot":
        ego_num_workers = 0
        robot_num_workers = args.num_workers
    elif iter_type == "egoexo":
        ego_num_workers = args.num_workers
        robot_num_workers = 0
    elif iter_type == "both":
        if cfg.training.load_lerobot_into_memory:
            ego_num_workers = args.num_workers
            robot_num_workers = 0
            args.accumulation_steps = args.accumulation_steps
        else:
            ego_num_workers = args.num_workers // 2
            robot_num_workers = args.num_workers // 2
            args.accumulation_steps = args.accumulation_steps // 2
    """
    End worker distribution logic
    """

    """
    Start dataloading logic
    """
    if iter_type in ["egoexo", "both"]:
        egoexo_dataloader = DataLoader(
            # subset_dataset if USE_SUBSET else dataset,
            # source_dataset,
            egoexo_train_dataset,
            batch_size=1,  # Because each video is a sample, and each CPU worker should only decode one video at a time
            shuffle=True,  # Keep unshuffled as requested
            num_workers=ego_num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if ego_num_workers > 0 else None,
            persistent_workers=True if ego_num_workers > 0 else None
        )

        egoexo_val_dataloader = DataLoader(
            # source_dataset,
            egoexo_val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0, # TODO: set this to a more special value... if the val is huge, this may start having an impact
            pin_memory=True,
            drop_last=False,
            # prefetch_factor=2 if args.num_workers > 1 else None,
            prefetch_factor=None,
            persistent_workers=None
        )
    else:
        egoexo_dataloader = None
        egoexo_val_dataloader = None

    # robot dataloader below
    # if iter_type in ["robot", "both"]:
    #     lerobot_train_dataloader = DataLoader(
    #         lerobot_train_dataset,
    #         batch_size=args.num_subclips_per_video, # accumulation steps * num_subclips_per_video defines the batch size for the egoexo dataloader. since we also accumulate robot batch per timestep, this should just be num subclips per video so that the lerobot and egoexo batch size are equivalent.
    #         shuffle=True,
    #         num_workers=robot_num_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #         prefetch_factor=2 if robot_num_workers > 0 else None,
    #         persistent_workers=True if robot_num_workers > 0 else None
    #     )
    # else:
    #     lerobot_train_dataloader = None
    """
    End dataloading logic
    """

    args.rank += gpu
    # torch.distributed.init_process_group(
    #     backend='nccl', init_method=args.dist_url,
    #     world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # TODO: commented out for debugging
    # debug = True
    # model = BarlowTwins(args).cuda(gpu)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # param_weights = []
    # param_biases = []
    # for param in model.parameters():
    #     if param.ndim == 1:
    #         param_biases.append(param)
    #     else:
    #         param_weights.append(param)
    # parameters = [{'params': param_weights}, {'params': param_biases}]
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
    #                  weight_decay_filter=True,
    #                  lars_adaptation_filter=True)

    # # automatically resume from checkpoint if it exists
    # if (args.checkpoint_dir / 'checkpoint.pth').is_file():
    #     ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
    #                       map_location='cpu')
    #     start_epoch = ckpt['epoch']
    #     model.load_state_dict(ckpt['model'])
    #     optimizer.load_state_dict(ckpt['optimizer'])
    # else:
    #     start_epoch = 0

    # dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    # # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=per_device_batch_size, num_workers=args.workers,
    #     pin_memory=True, sampler=sampler)

    # start_time = time.time()
    # scaler = torch.amp.GradScaler()
    
    start_epoch = 0
    lerobot_train_dataloader = None

    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)

        # we collect all the batches into a list,
        # then collate them after % accumulation_steps
        valid_batch_counter = 0

        for batch_idx, batch in enumerate(iter_util.iter_batches(iter_type, egoexo_dataloader, lerobot_train_dataloader, cfg)):
            """
            Start batch collation logic.
                We parallelize batch retrieval in the num_workers, then collate them after % accumulation_steps
            """
            if valid_batch_counter % args.accumulation_steps == 0:
                batch_queue = []            
                batch_queue_robot = []
                valid_batch_counter = 0

            if (valid_batch_counter + 1) % args.accumulation_steps == 0:
                if iter_type == "egoexo" or iter_type == "both":
                    """
                    The below logic:
                        Collates human
                        Collates robot
                        Collates both
                    """
                    batch_queue.append(batch["egoexo"])
                    # time the egoexo batch collate
                    start_time = time.perf_counter()

                    egoexo_batch = train_util.collate_batches_egoexo_dataset(batch_queue, None, expand_to_match_two_hands=False, index_with_bbox_presence=False)
                    end_time = time.perf_counter()
                    print(f"Egoexo batch collate time: {end_time - start_time} seconds")
                    # print(f"Time to first batch start: {time.perf_counter() - time_to_first_batch_start} seconds")

                    time_to_first_batch_start = time.perf_counter()

                    # if cfg.training.load_lerobot_into_memory: 
                    #     ep_indices = np.random.choice(np.arange(lerobot_train_dataset.num_episodes), size=lerobot_batch_size, replace=True)

                    #     # from: inclusive
                    #     # to: exclusive
                    #     lerobot_sampled_indices = np.random.randint(lerobot_train_dataset.episode_data_index["from"][ep_indices], 
                    #     lerobot_train_dataset.episode_data_index["to"][ep_indices]-cfg.horizon)

                    #     robot_batch = {
                    #         "observation.image.ego_global": torch.from_numpy(lerobot_observation_image_ego_global[lerobot_sampled_indices]), # B, H, W, C
                    #         "action": torch.from_numpy(lerobot_action[lerobot_sampled_indices[:, np.newaxis] + np.arange(cfg.horizon)[np.newaxis, :]]), # B, H, 10
                    #         "observation.state": torch.from_numpy(lerobot_observation_state[lerobot_sampled_indices[:, np.newaxis] + np.arange(cfg.n_obs_steps)[np.newaxis, :]]) # B, H, 10
                    #     }

                if iter_type == "robot" or iter_type == "both":
                    if cfg.training.load_lerobot_into_memory:
                        robot_batch = batch["robot"]
                    else:
                        batch_queue_robot.append(batch["robot"])

                        # time the robot batch collate

                        start_time = time.perf_counter()
                        robot_batch = collate_batches_lerobot_dataset(batch_queue_robot)

                        robot_batch['observation.image.ego_global'] = (robot_batch['observation.image.ego_global'].permute(0, 2, 3, 1) * 255).to(torch.uint8)

                        end_time = time.perf_counter()
                        print(f"Robot batch collate time: {end_time - start_time} seconds")


                # batch_to_use contains data from all embodiments (human and robot)
                batch_to_use = {}
                images_to_use = []
                
                if iter_type == "egoexo" or iter_type == "both":
                    images_to_use.append(egoexo_batch["cached_image"])
                if iter_type == "robot" or iter_type == "both":
                    # -> B, C, H, W -> B, H, W, C
                    images_to_use.append(robot_batch['observation.image.ego_global'])

                print("ln609 timing")
                start_time = time.perf_counter()

                # cached_image has the combined images from all embodiments
                batch_to_use["cached_image"] = torch.cat(images_to_use, dim=0)

                valid_batch_counter += 1
                print(f"ln615 time: {time.perf_counter() - start_time}")
            else:
                if iter_type == "egoexo" or iter_type == "both":
                    batch_queue.append(batch["egoexo"])
                if iter_type == "robot" or iter_type == "both":
                    batch_queue_robot.append(batch["robot"])
                valid_batch_counter += 1
                continue

            # """
            # Domain randomization logic
            # """
            # print("ln624 timing")
            # start_time_ln624 = time.perf_counter()
            # if cfg.get("colorjitter_augmentation", True):
            #     # B, H, W, C -> B, H, W, C
            #     # move to gpu so its using gpu color jitter
            #     # we move images to the GPU first because it does GPU colorjitter, drastically sped up
            #     batch_to_use["cached_image"] = colorjitter_augmentation(batch_to_use["cached_image"].to(next(model.parameters()).device))

            # if cfg.get("random_translation_augmentation", True):
            #     # B, H, W, C -> B, C, H, W, C -> B, H, W, C
            #     batch_to_use["cached_image"] = _RANDOM_TRANSLATION(batch_to_use["cached_image"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # end_time_ln676 = time.perf_counter()
            # print(f"ln676 to ln624 time: {end_time_ln676 - start_time_ln624}")
            # """
            # End domain randomization logic
            # """

            """
            Pull actions wrt to cam from batch
            """
            if iter_type == "egoexo" or iter_type == "both":
                # TODO: this is temporary because we still load all the batches even if the embodiment loss weight is 0
                ln714_time = time.perf_counter()
                # egoexo_batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in egoexo_batch.items()}
                print(f"ln714 time: {time.perf_counter() - ln714_time}")
                # model predict action does forward rollouts
                # need to convert world kp to camera frame kp
                # -> batch_size, 1, 21*3 -> batch_size, 1, 63
                ln720_time = time.perf_counter()
                egoexo_nproprio = train_util.get_nproprio_from_batch(egoexo_batch, torch.bfloat16, args.accumulation_steps * cfg.egoexo_dataset.num_subclips_per_video, 'cpu', egoexo_source_dataset.proprio_min_wrt_cam_running, egoexo_source_dataset.proprio_max_wrt_cam_running, cfg.training.egoexo_action_type).to(torch.bfloat16)
                print(f"ln720 time: {time.perf_counter() - ln720_time}")
                # -> batch_size, horizon-1, 21*3 
                ln723_time = time.perf_counter()
                egoexo_nactions = train_util.get_nactions_from_batch(egoexo_batch, torch.bfloat16, args.accumulation_steps * cfg.egoexo_dataset.num_subclips_per_video, 'cpu', egoexo_source_dataset.action_min_wrt_cam_running, egoexo_source_dataset.action_max_wrt_cam_running, cfg.training.egoexo_action_type).to(torch.bfloat16)
                print(f"ln723 time: {time.perf_counter() - ln723_time}")

            if iter_type == "robot" and cfg.training.load_lerobot_into_memory:
                # if we doing robot only, the egoexo batch  is None (but it should be set Before?)
                egoexo_batch = None

            import pdb; pdb.set_trace()



    # TODO: have to fix training loop
        # for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
    #         y1 = y1.cuda(gpu, non_blocking=True)
    #         y2 = y2.cuda(gpu, non_blocking=True)
    #         adjust_learning_rate(args, optimizer, loader, step)
    #         optimizer.zero_grad()
    #         with torch.amp.autocast():
    #             loss = model.forward(y1, y2)
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         if step % args.print_freq == 0:
    #             if args.rank == 0:
    #                 stats = dict(epoch=epoch, step=step,
    #                              lr_weights=optimizer.param_groups[0]['lr'],
    #                              lr_biases=optimizer.param_groups[1]['lr'],
    #                              loss=loss.item(),
    #                              time=int(time.time() - start_time))
    #                 print(json.dumps(stats))
    #                 print(json.dumps(stats), file=stats_file)
    #     if args.rank == 0:
    #         # save checkpoint
    #         state = dict(epoch=epoch + 1, model=model.state_dict(),
    #                      optimizer=optimizer.state_dict())
    #         torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    # if args.rank == 0:
    #     # save final model
    #     torch.save(model.module.backbone.state_dict(),
    #                args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.vision_backbone.fc = nn.Identity()
        self.language_action_backbone = LanguageActionEncoder(args)

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.vision_backbone(y1))

        z2 = self.projector(self.language_action_backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
