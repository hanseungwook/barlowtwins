# loads the annotations from the egoexo json directory
# usage:
# 1. iterate over the rgb frames
# 2. run bbox estimator, with low threshold so we get them all
# 3. load bbox in the dataset then we can train

# note: the egoexo annotations and the torchcodec are both 0 indexed, so we can feel free to use the same index each

# note: when we do the bbox finetuning even with the exo images, we store the keypoints under the ego keypoints key for simplicity

import sys
sys.path.append('../WiLoR-mini/')
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
import sys
sys.path.append('../../WiLoR-mini/')
import tqdm
import time
from torchvision.transforms.functional import to_pil_image

# we might not need this
import ego_utils.visualization_util as visualization_util
from dataset_util import get_ego_bounding_boxes_closest_to_keypoints, get_image_patches_from_bboxes, get_keypoints_in_patch, is_valid_take, get_egocam_name_for_take, get_exocam_names_for_take
import math
from train.train_util import colorjitter_augmentation
from torchcodec.decoders import VideoDecoder
from wilor_mini.pipelines.pipeline_util import build_detection_dict_from_yolo_results, extract_single_bbox_data_from_dict, build_detection_dict_from_batch, build_detection_dict_from_hands23_results
import pandas as pd
from data_processing.bbox_scale_util import MLPWrapper, make_bbox_features_from_hwc, make_bbox_features_from_x0y0x1y1, make_x0y0x1y1_from_hwc, make_bbox_centers_from_x0y0x1y1
from camera_util import process_aria_cam_wrt_world, process_exo_cam_wrt_world
from hands23.hands23_util import results_from_hands23
import random
from torchvision.transforms.functional import to_pil_image
from camera_util import check_np_corrupted

# Same ordering you used when writing the XML
JOINTS = [
    "Wrist",
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4",
]
JOINT2IDX = {name: i for i, name in enumerate(JOINTS)}

CHIRALITY2IDX = {"Left Hand": 0, "Right Hand": 1}

def scale_bbox(bbox, factor):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * factor, (y1 - y0) * factor
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def check_all_2d_kp_in_scaled_bbox(scaled_x0y0x1y1, kp_2d_fullimg_tensor, keypoint_present_tensor):
    """
    scaled_x0y0x1y1: (4,)
    kp_2d_fullimg_tensor: (21, 2)
    keypoint_present_tensor: (21,)
    """
    for kp_2d_fullimg, keypoint_present in zip(kp_2d_fullimg_tensor, keypoint_present_tensor):
        if keypoint_present and (kp_2d_fullimg[0] < scaled_x0y0x1y1[0] or kp_2d_fullimg[0] > scaled_x0y0x1y1[2] or kp_2d_fullimg[1] < scaled_x0y0x1y1[1] or kp_2d_fullimg[1] > scaled_x0y0x1y1[3]):
            return False
    return True

class EgoExoDatasetFromGTJson(Dataset):
    """
    Dataset for hand keypoint training.
    Each element is a video.
    Expected data format:
    - images: directory with images
    - annotations: JSON file with format:
      {
        "image_name.jpg": {
          "hands": [
            {
              "bbox": [x1, y1, x2, y2],
              "keypoints_2d": [[x1, y1], [x2, y2], ...], # 21 keypoints
              "is_right": True/False,
              "visible": [1, 1, 0, ...] # optional visibility flags
            }
          ]
        }
      }
    """
    
    def __init__(self, release_dir, 
                 annotation_root, joint_wrt_cam_cache_dir=None, bbox_json_root=None, 
                 bbox_ego_npy_filepath=None, 
                 bbox_exo_npy_filepath=None,
                 bbox_npy_root=None,
                 three_d_keypoints_torch_root=None,
                  image_patch_size=256, rescale_factor=2.5,
                 render_debug_image=False,
                 original_image_size=512,
                 original_focal=150,
                 new_focal=340,
                 num_subclips_per_video=1,
                 render_debug_frames=False,
                 debug_max_takes=1e10,
                ignore_take_names=[],
                # num_workers=1,
                prefetch_factor=2,
                return_full_images=False,
                allow_takes_with_annotations_only=False,
                allowed_parent_tasks=["Cooking", "Bike Repair"],
                allowed_task_names=[],
                allowed_take_names=[],
                early_return_rectified_frames=False,
                frames_type_to_use=None,
                partition_size=200,
                hand_detector=None,
                cam_type_to_use="ego",
                right_hand_bbox_scaler=None,
                left_hand_bbox_scaler=None,
                colorjitter_augmentation=True,
                loop_over_cameras_first=True,
                horizon=31,
                three_d_fps=10,
                cached_rgb_dir=None,
                filter_clip_proprio_norm=None,
                has_bbox_npy_bool=False,
                bbox_model_to_use="hands23",
                rectified_ego_focal_length=411,
                load_cam_data=True):
                #  ignore_take_names=["cmu_bike01_2"]):
        # assert num_workers > 0
        # assert prefetch_factor >= 2
        """
        keypoint_image_size: size of the image the keypoints were labelled on
        keypoint_focal: focal length of the camera used to render the keypoints
        image_focal: focal length of the camera used to render the image
        cache_dir: used to cache the preprocessing in init, which may take a long time
        iterate_over_frames: for every annotation we have, iterate over the frame. Useful for rendering out the whole dataset.
        frames_type_to_use: whether to display all frames or only frames with annotations. If "all_frames", partition the frames based on batch size.
        partition_size: size of the partition to use for frames_type_to_use == "all_frames". Partition meaning we split the frames for videos into a max partition size per batch.
        three_d_fps: fps of the 3d keypoints. If we have a single image, it's okay to decode at any fps, and use a separate three_d_fps. Otherwise they should be the same.
        filter_clip_proprio_norm: if not None, filter the clips based on the norm of the proprioception (we want to ignore identities).
        return_full_img: if True, process the mp4s into the original resolution (large) image. Otherwise, we only have the cached image
        """
        if return_full_images:
            from vrs_util import get_fisheye_rgb_camera_calibration, undistort_aria_given_device_calib, get_gopro_calibration, undistort_exocam

        self.cached_rgb_dir = cached_rgb_dir
        self.has_bbox_npy_bool = has_bbox_npy_bool

        assert cam_type_to_use in ["ego", "exo", "all"]
        print("Using debug max takes", debug_max_takes)
        self.cam_type_to_use = cam_type_to_use
        self.bbox_ego_npy_filepath = bbox_ego_npy_filepath
        self.bbox_exo_npy_filepath = bbox_exo_npy_filepath
        self.bbox_npy_root = bbox_npy_root
        # assert only bbox_npy_root can be set or both bbox_ego_npy_filepath and bbox_exo_npy_filepath can be set
        # assert (bbox_npy_root is not None and bbox_ego_npy_filepath is None and bbox_exo_npy_filepath is None) or (bbox_npy_root is None and (bbox_ego_npy_filepath is not None or bbox_exo_npy_filepath is not None))
        
        self.three_d_keypoints_torch_root = three_d_keypoints_torch_root
        self.release_dir = release_dir
        self.take_root = os.path.join(release_dir, "takes")
        self.annotation_root = annotation_root
        self.image_patch_size = image_patch_size
        # self.image_original_size = 1404
        self.rescale_factor = rescale_factor
        
        self.new_focal = new_focal

        self.joint_wrt_cam_cache_dir = joint_wrt_cam_cache_dir
        if joint_wrt_cam_cache_dir is not None:
            os.makedirs(joint_wrt_cam_cache_dir, exist_ok=True)
        
        annotation_dir = os.path.join(release_dir, "annotations/")  # annotation folder
        egopose_ann_dir = os.path.join(
            annotation_dir, f"ego_pose/train/hand/annotation"
        )

        self.egopose_ann_dir = egopose_ann_dir
        self.loop_over_cameras_first = loop_over_cameras_first

        # this is action horizon + proprioception single frame
        # in pizero, the horizon is just the action horizon
        self.horizon = horizon
        self.three_d_fps = three_d_fps

        # process annotations
        # for every take name, 

        takes_json_lst = json.load(open(os.path.join(release_dir, "takes.json")))

        self.take_name_to_take = {take["take_name"]: take for take in takes_json_lst}

        self.selected_take_names = []

        self.take_name_to_video_decoder = {}
        self.take_name_to_egocam_video_path = {}

        self.take_name_to_egocam_video_path_cached = {}
        self.take_name_to_exocams_video_path = {}

        self.take_name_to_egocam_name = {}
        self.take_name_to_exocams_names = {}

        self.take_name_to_egocam_fisheye_rgb_calibration = {}
        self.take_name_to_exocams_calibration_parameters = {}

        self.allowed_parent_tasks = allowed_parent_tasks
        self.allowed_task_names = allowed_task_names
        self.allowed_take_names = allowed_take_names

        self.allow_takes_with_annotations_only = allow_takes_with_annotations_only
        self.ignore_take_names = ignore_take_names

        self.colorjitter_augmentation = colorjitter_augmentation

        self.load_cam_data = load_cam_data


        """
        Process extrinsics
        """
        self.take_name_to_aria_cam_wrt_world_save_path = {}

        self.take_name_to_exo_cam_wrt_world_save_path = {}
        self.take_name_to_exo_cam_intrinsics_save_path = {}

        count = 0

        # print("ln151")
        is_valid_take("iiith_cooking_134_2", self.take_name_to_take, self.allowed_parent_tasks, self.allowed_task_names, self.annotation_root, self.allow_takes_with_annotations_only, self.ignore_take_names, self.allowed_take_names, self.has_bbox_npy_bool, rectified_ego_focal_length, self.three_d_keypoints_torch_root, self.cached_rgb_dir)
        for take_name in tqdm.tqdm(sorted(self.take_name_to_take.keys())):           
            if is_valid_take(take_name, self.take_name_to_take, self.allowed_parent_tasks, self.allowed_task_names, self.annotation_root, self.allow_takes_with_annotations_only, self.ignore_take_names, self.allowed_take_names, self.has_bbox_npy_bool, rectified_ego_focal_length, self.three_d_keypoints_torch_root, self.cached_rgb_dir):
                print("Preparing take", take_name)

                # TODO: filter based on cam type?
                try:
                    # the ego and exo must both process for the file to be added
                    ego_camwrtworld_save_path, _ = process_aria_cam_wrt_world(self.release_dir, take_name, export_name=f"aria_cam_wrt_world_closedloop_poses_30fps.npy", decode_fps=30, recreate_if_exists=False)

                    if check_np_corrupted(ego_camwrtworld_save_path):
                        print(f"Ego cam wrt world is corrupted for {take_name}, skipping")
                        continue

                    self.take_name_to_aria_cam_wrt_world_save_path[take_name] = ego_camwrtworld_save_path

                    self.take_name_to_exocams_names[take_name] = get_exocam_names_for_take(release_dir, take_name, self.take_name_to_take)

                    self.take_name_to_exo_cam_wrt_world_save_path[take_name] = {}
                    self.take_name_to_exo_cam_intrinsics_save_path[take_name] = {}
                    for exo_cam_name in self.take_name_to_exocams_names[take_name]:
                        exo_camwrtworld_save_path, exo_intrinsics_save_path, _, _, _ = process_exo_cam_wrt_world(self.release_dir, take_name, exo_cam_name, distorted_image_size=(2160, 3840), rectified_image_size=(2160, 3840), recreate_if_exists=False)
                        self.take_name_to_exo_cam_wrt_world_save_path[take_name][exo_cam_name] = exo_camwrtworld_save_path
                        self.take_name_to_exo_cam_intrinsics_save_path[take_name][exo_cam_name] = exo_intrinsics_save_path
                except Exception as e:
                    print(f"Error processing aria cam wrt world for {take_name}: {e}, skipping")
                    continue

                try:
                    """
                    For each camera, 
                    - get the video path
                    - get the camera calibration
                    """
                    self.take_name_to_egocam_name[take_name] = get_egocam_name_for_take(take_name, self.take_name_to_take)

                    if self.cam_type_to_use == "ego" or self.cam_type_to_use == "all":
                        ego_cam_name = self.take_name_to_egocam_name[take_name]

                        if cached_rgb_dir:
                            video_name = f"224x224_focal{rectified_ego_focal_length}_rectified_video.mp4"
                            self.take_name_to_egocam_video_path_cached[take_name] = os.path.join(cached_rgb_dir, 
                                                                                    take_name, 
                                                                                    video_name)
                        if return_full_images:
                            self.take_name_to_egocam_video_path[take_name] = os.path.join(self.take_root, 
                                                                                    take_name, 
                                                                                    self.take_name_to_take[take_name]['frame_aligned_videos'][ego_cam_name]['rgb']['relative_path'])
                            self.take_name_to_egocam_fisheye_rgb_calibration[take_name] = get_fisheye_rgb_camera_calibration(os.path.join(release_dir, "takes", take_name, f"{ego_cam_name}_noimagestreams.vrs"))


                    if self.cam_type_to_use == "exo" or self.cam_type_to_use == "all":
                        self.take_name_to_exocams_video_path[take_name] = {}
                        self.take_name_to_exocams_calibration_parameters[take_name] = {}
                        for exo_cam_name in self.take_name_to_exocams_names[take_name]:
                            self.take_name_to_exocams_video_path[take_name][exo_cam_name] = os.path.join(self.take_root, 
                                                                                                        take_name,
                                                                                                        self.take_name_to_take[take_name]['frame_aligned_videos'][exo_cam_name]['0']['relative_path'])
                            
                            if not cached_rgb_dir:
                                self.take_name_to_exocams_calibration_parameters[take_name][exo_cam_name] = get_gopro_calibration(release_dir, take_name, exo_cam_name)
                except Exception as e:
                    print(f"Error processing exo cams for {take_name}: {e}, skipping")
                    continue

                # if both intrinsics and extrinsics can be processed, add the take
                # if we don't skip above, then add the take
                self.selected_take_names.append(take_name)


                count += 1
                if count > debug_max_takes:
                    print("breaking early")
                    break
        assert count > 0                        
        # if self.cache_dir is not None:
        #     # save the take_name_to_fisheye_rgb_camera_calibration to a pickle file
        #     with open(os.path.join(self.cache_dir, "take_name_to_fisheye_rgb_camera_calibration.pkl"), "wb") as f:
        #         pickle.dump(self.take_name_to_fisheye_rgb_camera_calibration, f)

        # assert len(self.take_names_with_annotations) > 0, "No annotations found"

        self.take_idx_to_take_name = {i: take_name for i, take_name in enumerate(self.selected_take_names)}
        self.take_name_to_take_idx = {take_name: i for i, take_name in enumerate(self.selected_take_names)}

   
        # # if path ends in json, load json
        # if annotation_file.endswith('.json'):
        #     # note: the json is the egoexo json
        #     with open(annotation_file, 'r') as f:
        #         self.raw_annotations = json.load(f)
        #     # Create list of all hand instances
        #     self.samples = []
        #     for image_name, data in self.annotations.items():
        #         if 'hands' in data:
        #             for hand_idx, hand_data in enumerate(data['hands']):
        #                 self.samples.append({
        #                     'image_name': image_name,
        #                     'hand_idx': hand_idx,
        #                     'hand_data': hand_data
        #                 })
        # if path ends in xml, load xml
            # parse cvat xml
            # filepaths: num_els
            # xy: num_els x 2 x 21 x 2
            # outside: num_els x 2 x 21
            # self.filepaths, self.xy, self.outside = from_cvat_xml(annotation_file)

            # # filter filepaths, only include samples with at least one hand visible
            # # outside means NOT VISIBLE
            # valid_indices = np.sum(self.outside, axis=(1, 2)) <= 21

            # self.filepaths = np.array(self.filepaths)[valid_indices]
            # self.xy = self.xy[valid_indices]
            # self.outside = self.outside[valid_indices]
            # self.label_type = "cvat"

            # self.bbox_json_loaded = json.load(open(bbox_json_path))

            # # each filepath can have 1 or 2 bboxes.
            # # should wh
            # self.global_filepaths = []
            # self.global_xy = []
            # self.global_outside = []
            # self.global_hand_idx = []
            # self.global_bbox = []
            # global_counter = 0
            # samples_skip_count = 0
            # for filepath_local_idx, filepath in enumerate(self.filepaths):
            #     image_filename =  os.path.basename(filepath)
            #     image_name = image_filename.split('.')[0]

            #     # sort bboxes so "left_hand" is first and "right_hand" is second
            #     # also filter out hands without labels
            #     for hand_idx, hand_data in enumerate(sorted(self.bbox_json_loaded[image_name], key=lambda bbox_data: int(bbox_data[1] != "left_hand"))):

            #         if not np.isnan(self.xy[filepath_local_idx]).any() and not self.outside[filepath_local_idx][hand_idx][0]:
            #             joint_values = self.xy[filepath_local_idx][hand_idx]

            #             # if any joint value is < 0 or > the image size, raise exception
            #             if np.any(joint_values < -400) or np.any(joint_values > 1.5*self.image_original_size):
            #                 samples_skip_count += 1
            #                 print(f"Joint values way too large. Skipping {joint_values}")
            #             self.global_filepaths.append(filepath)
            #             self.global_xy.append(joint_values)
            #             self.global_outside.append(self.outside[filepath_local_idx][hand_idx])
            #             self.global_hand_idx.append(hand_idx)
            #             self.global_bbox.append(np.array(hand_data[0]))
            #             global_counter += 1

            # assert global_counter > 0, "No valid hands found"
        self.debug = True
        self.num_subclips_per_video = num_subclips_per_video
        self.render_debug_frames = render_debug_frames
        self.keypoints_map, self.skeleton, self.pose_kpt_color = visualization_util.get_hands_metadata()

        self.bbox_json_root = bbox_json_root
        self.debug_max_takes = debug_max_takes

        # self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.return_full_images = return_full_images
        self.early_return_rectified_frames = early_return_rectified_frames
        self.frames_type_to_use = frames_type_to_use 
        assert frames_type_to_use in ["iterate_all_frames", "at_least_one_bbox", "inflate_length", "valid_3d_kp_clips", None]
        self.partition_size = partition_size

        if self.frames_type_to_use == "iterate_all_frames" and not self.allow_takes_with_annotations_only:
            # we have a global array
            # video_idx_1_partition_1 ... video_idx_1_partition_n video_idx_2_partition_1 ... video_idx_2_partition_n ...

            # if we have multiple cameras, they get concatenated
            # (take1, cam1, partition1) (take1, cam1, partition2) ... (take1, cam1, partitionn) (take1, cam2, partition1) ... (take1, cam2, partitionn) ... (take2, cam1, partition1) ... (take2, cam2, partitionn) ...

            self.sample_idx_to_take_name = {}
            self.sample_idx_to_cam_name = {}
            self.sample_idx_to_partition_range = {}
            self.sample_idx_to_last_partition_boolean = {}

            sample_idx = 0

            # note: we are assuming all camera videos have exactly the same length, which should be satisfied by exoego
            for take_name in self.selected_take_names:
                take_num_frames = math.floor(self.take_name_to_take[take_name]['duration_sec'] * 30)

                cams_to_use = []
                cams_to_use += [self.take_name_to_egocam_name[take_name]] if self.cam_type_to_use == "ego" or self.cam_type_to_use == "all" else []
                cams_to_use += [exo_cam_name for exo_cam_name in self.take_name_to_exocams_names[take_name]] if self.cam_type_to_use == "exo" or self.cam_type_to_use == "all" else []

                if self.loop_over_cameras_first:
                    for cam_name in sorted(cams_to_use):
                        # take_num_frames = 69
                        # print(f"USING DEBUG TAKE NUM FRAMES: {take_num_frames}") 
                        for partition_idx in range(math.ceil(take_num_frames / self.partition_size)):
                            self.sample_idx_to_take_name[sample_idx] = take_name
                            self.sample_idx_to_cam_name[sample_idx] = cam_name
                            # (start_inclusive, stop_exclusive)
                            self.sample_idx_to_partition_range[sample_idx] = (partition_idx * self.partition_size, 
                                                                            min((partition_idx + 1) * self.partition_size, take_num_frames))
                            self.sample_idx_to_last_partition_boolean[sample_idx] = (partition_idx + 1) * self.partition_size >= take_num_frames
                            sample_idx += 1
                else:
                    for partition_idx in range(math.ceil(take_num_frames / self.partition_size)):
                        for cam_name in sorted(cams_to_use):
                            self.sample_idx_to_take_name[sample_idx] = take_name
                            self.sample_idx_to_cam_name[sample_idx] = cam_name
                            # (start_inclusive, stop_exclusive)
                            self.sample_idx_to_partition_range[sample_idx] = (partition_idx * self.partition_size, 
                                                                            min((partition_idx + 1) * self.partition_size, take_num_frames))
                            self.sample_idx_to_last_partition_boolean[sample_idx] = (partition_idx + 1) * self.partition_size >= take_num_frames
                            sample_idx += 1

        self.take_to_take_num_frames = {take_name: math.floor(self.take_name_to_take[take_name]['duration_sec'] * 30) for take_name in self.selected_take_names}
        self.hand_detector = hand_detector
        self.right_hand_bbox_scaler = right_hand_bbox_scaler
        self.left_hand_bbox_scaler = left_hand_bbox_scaler
        # self.cam_type = cam_type
        
        """
        Compute limit normalizers
            Loop over all 3D hands in camera frame, and get the min and max
            There is a single normalizer over all the data

        Note: the three d keypoints are stored in a meta folder labelled by take name. So we need to loop over this directory.

        Note: doing right hand only for now
        """

        self.filter_clip_proprio_norm = filter_clip_proprio_norm
        self.bbox_model_to_use = bbox_model_to_use
        self.rectified_ego_focal_length = rectified_ego_focal_length

    def set_normalization_from_take_names(self, take_names):
        # for take_name in take_names:
        #     assert take_name in self.selected_take_names, f"Take {take_name} not in selected take names"
        # x_min_wrt_cam_running = torch.ones(21, 3) * torch.inf
        # x_max_wrt_cam_running = torch.ones(21, 3) * -torch.inf

        x_min_wrt_cam_collected = []
        x_max_wrt_cam_collected = []

        if self.three_d_keypoints_torch_root is not None:
            for take_name in take_names:
                if os.path.exists(os.path.join(self.joint_wrt_cam_cache_dir, f"joints_wrt_cam_valid_{take_name}.pt")):
                    joints_wrt_cam_valid = torch.load(os.path.join(self.joint_wrt_cam_cache_dir, f"joints_wrt_cam_valid_{take_name}.pt"), weights_only=False)
                    # x_min_wrt_cam_running = torch.min(x_min_wrt_cam_running, torch.amin(joints_wrt_cam_valid, axis=[0, 1]))
                    # x_max_wrt_cam_running = torch.max(x_max_wrt_cam_running, torch.amax(joints_wrt_cam_valid, axis=[0, 1]))
                    x_min_wrt_cam_collected.append(joints_wrt_cam_valid)
                    x_max_wrt_cam_collected.append(joints_wrt_cam_valid)
                    continue
                
                three_d_keypoints_torch = torch.load(os.path.join(self.three_d_keypoints_torch_root, take_name, "serialization_compiled.pt"))
                # convert to cam first

                # then take min 
                aria_cam_wrt_world = torch.from_numpy(np.load(self.take_name_to_aria_cam_wrt_world_save_path[take_name])).to(torch.float32)

                # using as strided, create clip views of the data
                # normalize the clip views according to the 0th frame
                # then flatten the clips and take the min and max

                # -> num_frames, 21, 3 -> num_frames - H, H, 21, 3. NOTE YOU MUST USE THE CLONE OR THIS WILL FUCK UP
                joints_wrt_world_clips = three_d_keypoints_torch['nonlinear_results_slerped'][:, 1, :].clone().as_strided(size=(self.take_to_take_num_frames[take_name] - self.horizon, self.horizon, 21, 3), 
                                                                                                            stride=(21*3, 21*3, 3, 1))
                
                joints_wrt_world_homo = torch.cat([joints_wrt_world_clips, torch.ones((*joints_wrt_world_clips.shape[:3], 1))], axis=-1)
                # TODO: also filter... we only want slerp bool true frames!

                # -> num_frames - H, H, 21, 4
                joints_wrt_cam = torch.einsum("bhjd, bid -> bhji",  joints_wrt_world_homo, torch.linalg.inv(aria_cam_wrt_world)[:self.take_to_take_num_frames[take_name] - self.horizon])[..., :3]

                valid_keys_boolean = torch.sum(three_d_keypoints_torch['slerp_boolean'][:, 1].clone().as_strided(size=(self.take_to_take_num_frames[take_name]-self.horizon, self.horizon), stride=(1, 1)), axis=-1) == self.horizon

                joints_wrt_cam_valid = joints_wrt_cam[valid_keys_boolean]

                # serialize joints wrt cam valid to cache directory
                if self.joint_wrt_cam_cache_dir is not None:
                    torch.save(joints_wrt_cam_valid, os.path.join(self.joint_wrt_cam_cache_dir, f"joints_wrt_cam_valid_{take_name}.pt"))

                x_min_wrt_cam_collected.append(joints_wrt_cam_valid)
                x_max_wrt_cam_collected.append(joints_wrt_cam_valid)
                # x_min_wrt_cam_running = torch.min(x_min_wrt_cam_running, torch.amin(joints_wrt_cam_valid, axis=[0, 1]))
                # x_max_wrt_cam_running = torch.max(x_max_wrt_cam_running, torch.amax(joints_wrt_cam_valid, axis=[0, 1]))
            # self.x_min_wrt_cam_running = x_min_wrt_cam_running
            # self.x_max_wrt_cam_running = x_max_wrt_cam_running

            self.x_min_wrt_cam_running = torch.quantile(torch.cat(x_min_wrt_cam_collected, axis=0).reshape(-1, 21, 3), q=0.01, dim=0)
            self.x_max_wrt_cam_running = torch.quantile(torch.cat(x_max_wrt_cam_collected, axis=0).reshape(-1, 21, 3), q=.99, dim=0)

            assert torch.min(self.x_min_wrt_cam_running) > -2, "If our normalization statistics are less than -2 meters we have a problem"
            assert torch.max(self.x_max_wrt_cam_running) < 2, "If our normalization statistics are greater than 2 meters we have a problem"
        
    def get_joints_wrt_cam_valid_all(self):
        assert self.joint_wrt_cam_cache_dir is not None, "Cache dir not set"
        joints_wrt_cam_valid = {}
        for take_name in self.selected_take_names:
            joints_wrt_cam_valid[take_name] = torch.load(os.path.join(self.joint_wrt_cam_cache_dir, f"joints_wrt_cam_valid_{take_name}.pt"))
        return joints_wrt_cam_valid

    def __len__(self):
        if self.allow_takes_with_annotations_only:
            # assumign right now that there are small number of annotations per take
            return len(self.selected_take_names)
        if self.frames_type_to_use == "iterate_all_frames":
            return len(self.sample_idx_to_take_name.keys())
        else:
            return len(self.selected_take_names)
    
    def __getitem__(self, sample_idx):
        if self.frames_type_to_use == "iterate_all_frames" and not self.allow_takes_with_annotations_only:
            take_name = self.sample_idx_to_take_name[sample_idx]
            partition_start, partition_stop = self.sample_idx_to_partition_range[sample_idx]
            cam_name = self.sample_idx_to_cam_name[sample_idx]
            cam_type = "ego" if cam_name == self.take_name_to_egocam_name[take_name] else "exo"
        else:
            # each video corresponds to one sample
            # if our dataset is longer than num videos, start resampling videos
            # sample_idx = sample_idx % len(self.selected_take_names)
            take_name = self.take_idx_to_take_name[sample_idx]

            if self.cam_type_to_use == "ego":
                cam_name = self.take_name_to_egocam_name[take_name]
                cam_type = "ego"
            elif self.cam_type_to_use == "exo":
                # randomly sample a camera
                cam_name = random.choice(self.take_name_to_exocams_names[take_name])
                cam_type = "exo"
            else:
                raise ValueError(f"Camera type {self.cam_type_to_use} not supported")

        if bool(self.bbox_npy_root):
            bbox_json_for_take_path = os.path.join(self.bbox_npy_root, take_name, f"{self.bbox_model_to_use}_{cam_type}_compiled", "all_hands_found_across_frames.npy")

            bbox_json_for_take = np.load(bbox_json_for_take_path, allow_pickle=True).item()[take_name]
            # overwrite the cam name here, before it is used anywhere
            if not self.frames_type_to_use == "iterate_all_frames":
                cam_name = random.choice(list(bbox_json_for_take.keys()))
        elif bool(self.bbox_ego_npy_filepath) or bool(self.bbox_exo_npy_filepath):
            if cam_name == self.take_name_to_egocam_name[take_name]:
                bbox_json_for_take = np.load(self.bbox_ego_npy_filepath, allow_pickle=True).item()[take_name]
            else:
                bbox_json_for_take = np.load(self.bbox_exo_npy_filepath, allow_pickle=True).item()[take_name]
        else:
            # assert self.hand_detector is not None, "Bbox json root not set and hand detector not given"
            bbox_json_for_take = None
        take = self.take_name_to_take[take_name]


        if self.return_full_images:
            if cam_name == self.take_name_to_egocam_name[take_name]:
                video_decoder = VideoDecoder(self.take_name_to_egocam_video_path[take_name], num_ffmpeg_threads=0)
            else:
                assert cam_name in self.take_name_to_exocams_video_path[take_name], f"Camera {cam_name} not in ego or exo cam list for {take_name}"
                video_decoder = VideoDecoder(self.take_name_to_exocams_video_path[take_name][cam_name], num_ffmpeg_threads=0)

                # Height, Width
            full_image_size = (video_decoder.metadata.height, video_decoder.metadata.width)

        if self.cached_rgb_dir:
            cached_video_decoder = VideoDecoder(self.take_name_to_egocam_video_path_cached[take_name], num_ffmpeg_threads=0)
            # cached_full_image_size = (cached_video_decoder.metadata.height, cached_video_decoder.metadata.width)


        # if self.bbox_json_root is not None:
        #     t0 = time.time()
        #     bbox_json_for_take = json.load(open(os.path.join(self.bbox_json_root, take_name, "bbox.json")))
        #     print(f"time taken to load bbox json: {time.time() - t0}")


        if self.three_d_keypoints_torch_root is not None:
            # TODO: note that this will be for the specific take
            three_d_keypoints_torch = torch.load(os.path.join(self.three_d_keypoints_torch_root, take_name, "serialization_compiled.pt"))
            # three_d_keypoints_torch = three_d_keypoints_torch[take_name]
        else:
            three_d_keypoints_torch = None

        # load annotations
        annotation_json_path = os.path.join(self.egopose_ann_dir, take['take_uid'] + ".json")
        if os.path.exists(annotation_json_path) and self.allow_takes_with_annotations_only:
            t0 = time.time()
            annotation_json = json.load(open(annotation_json_path))
            print(f"time taken to load annotations: {time.time() - t0}")

            # get all frame indices that actually have annotations (some are empty).
            # the 0 is not indexing into the camera, it's just because for some reason annotation_json[k] is always a list of length 1
            annotation_key_list = [int(k) for k in annotation_json.keys() if bool(annotation_json[k][0]['annotation2D'])]

            valid_key_list = set(annotation_key_list).intersection(set(bbox_json_for_take[cam_name]['all_hands_found_bboxes_across_frames'].keys()))
            valid_keys = torch.Tensor(list(valid_key_list)).int()

            # TODO: this is a hack to get the frames with annotations
            # sample valid_keys with replacement
            if self.frames_type_to_use == "iterate_all_frames":
                sampled_frame_idxs_30fps = valid_keys
            else:
                if len(valid_keys) == 0:
                    # make empty tensor here
                    sampled_frame_idxs_30fps = torch.zeros((0,), dtype=torch.int32)
                else:
                    # do the same as above line with torch multinomial
                    sampled_frame_idxs_30fps_localidxs = torch.multinomial(torch.ones(len(valid_keys)), self.num_subclips_per_video, replacement=True)
                    sampled_frame_idxs_30fps = valid_keys[sampled_frame_idxs_30fps_localidxs]
        else:
            # sample arbitrary frames in the video at 30fps
            # TODO: actually we should sample frames where we found a bounding box
            if self.frames_type_to_use == "iterate_all_frames":
                sampled_frame_idxs_30fps = torch.arange(partition_start, partition_stop)
            elif self.frames_type_to_use == "valid_3d_kp_clips":
                # we can only sample images where the future horizon actions and current actions are valid
                # note: let's only train on right hands for now because there is the most movement

                # -> num_frames - H, H
                slerp_bool_strided = three_d_keypoints_torch['slerp_boolean'][:, 1].clone().as_strided(size=(self.take_to_take_num_frames[take_name]-self.horizon, self.horizon), stride=(1, 1))

                # -> num_frames - H, H, 21, 3
                nonlinear_joints_strided = three_d_keypoints_torch['nonlinear_results_slerped'][:, 1, ...].clone().as_strided(size=(self.take_to_take_num_frames[take_name]-self.horizon, self.horizon, 21, 3), stride=(21*3, 21*3, 3, 1))

                # -> num_frames - H
                nonlinear_joints_norm_strided = torch.linalg.norm(nonlinear_joints_strided[:, 1:, ...] - nonlinear_joints_strided[:, 0:1, ...], axis=[-1, -2]).mean(axis=-1)


                # only keep clips where every frame in the horizon is present
                # -> num_frames - H
                horizon_bool = torch.sum(slerp_bool_strided, axis=-1) == self.horizon
                norm_bool = nonlinear_joints_norm_strided >= self.filter_clip_proprio_norm

                valid_keys_boolean = horizon_bool * norm_bool

                valid_keys = torch.arange(self.take_to_take_num_frames[take_name]-self.horizon)[valid_keys_boolean]
                sampled_frame_idxs_30fps_localidxs = torch.multinomial(torch.ones(len(valid_keys)), self.num_subclips_per_video, replacement=True)
                sampled_frame_idxs_30fps = valid_keys[sampled_frame_idxs_30fps_localidxs]

                # from the sampled frames, they need to at least not be zero hands
                assert not torch.any(torch.sum(three_d_keypoints_torch['nonlinear_results_slerped'][:, 1, ...][sampled_frame_idxs_30fps] == 0, axis=[1, 2]))
                assert not torch.any(torch.sum(three_d_keypoints_torch['nonlinear_results_slerped'][:, 1, ...][sampled_frame_idxs_30fps[:, None] + torch.arange(self.horizon)[None, :]], axis=[2, 3]) == 0)
            else:
                raise NotImplementedError("Frames type to use not implemented")


        # if 4019 in sampled_frame_idxs_30fps:
        #     import pdb; pdb.set_trace()
        num_sampled_frames_for_this_video = sampled_frame_idxs_30fps.shape[0]
        num_placeholder_frames = num_sampled_frames_for_this_video
        # sampled_frame_idxs_30fps = np.random.choice(valid_keys, size=(1), replace=True)

        # decode_start =time.time()
        # -> inflate_length, C, H, W
        if self.return_full_images:
            print(f"decoding video for take {take_name} and cam {cam_name}")
            print(f"sampled_frame_idxs_30fps: {sampled_frame_idxs_30fps[:10]}")
            t0 = time.time()
            fisheye_rgb_data = video_decoder.get_frames_at(sampled_frame_idxs_30fps.tolist()).data
            print(f"time taken to decode video: {time.time() - t0}")

            # -> inflate_length, H, W, C
            cv2_fisheye_rgb_data = fisheye_rgb_data.permute(0, 2, 3, 1).cpu().numpy()

        if self.cached_rgb_dir:
            cached_fisheye_rgb_data = cached_video_decoder.get_frames_at(sampled_frame_idxs_30fps.tolist()).data
            # -> inflate_length, H, W, C
            cv2_cached_rectified_rgb_data = cached_fisheye_rgb_data.permute(0, 2, 3, 1).cpu().numpy()

        t0 = time.time()

        if self.return_full_images:
            rgb_frames = torch.zeros((num_placeholder_frames, full_image_size[0], full_image_size[1], 3), dtype=torch.uint8)
            rgb_debug_frames = torch.zeros((num_placeholder_frames, full_image_size[0], full_image_size[1], 3), dtype=torch.uint8)
        rgb_patches_tensor = torch.zeros((num_placeholder_frames, 2, self.image_patch_size, self.image_patch_size, 3), dtype=torch.uint8)
        
        ego_bboxes_tensor = torch.zeros((num_placeholder_frames, 2, 4), dtype=torch.float32)
        ego_bboxes_present_tensor = torch.zeros((num_placeholder_frames, 2), dtype=torch.bool)

        ego_bboxes_centers = torch.zeros((num_placeholder_frames, 2, 2), dtype=torch.float32)
        ego_bboxes_sizes = torch.zeros((num_placeholder_frames, 2), dtype=torch.float32)

        ego_bboxes_confidences_tensor = torch.zeros((num_placeholder_frames, 2), dtype=torch.float32)

        ego_keypoint_2d_fullimg_tensor = torch.zeros((num_placeholder_frames, 2, 21, 2), dtype=torch.float32)

        # n_length, horizon-1 (1 for image), 21 (one hand for now), 3
        ego_keypoint_3d_wrtworld_tensor = torch.zeros((num_placeholder_frames, self.horizon, 2, 21, 3), dtype=torch.float32)

        # after we use linear interpolation, most (90%) of the video should be filled
        # the sampling function should not sample any illegal indices
        ego_keypoint_3d_present_tensor = torch.zeros((num_placeholder_frames, self.horizon, 2), dtype=torch.bool)
        if self.three_d_keypoints_torch_root is not None:
            # we should probably save this as a tensor and not dict keys
            # three_d_keypoints_torch[take_name][9759][0]['nonlinear_results'][0]
            ego_keypoint_3d_wrtworld_tensor[:, :, 1, :, :] = (three_d_keypoints_torch['nonlinear_results_slerped'][:, 1, ...][sampled_frame_idxs_30fps[:, None] + torch.arange(self.horizon)[None, :]])
            # ego_keypoint_3d_present_tensor[:, :, 1] = torch.ones((num_placeholder_frames, 2), dtype=torch.bool)


        keypoint_present_tensor = torch.zeros((num_placeholder_frames, 2, 21), dtype=torch.bool)
        ego_keypoint_2d_patches_tensor = torch.zeros((num_placeholder_frames, 2, 21, 2), dtype=torch.float32)


        """
        Rectify frames 
        """
        t0 = time.time()
        if self.cached_rgb_dir:
            rgb_cached_frames = cv2_cached_rectified_rgb_data
        if self.return_full_images:
            from vrs_util import get_fisheye_rgb_camera_calibration, undistort_aria_given_device_calib, get_gopro_calibration, undistort_exocam

            for local_frame_idx in range(num_sampled_frames_for_this_video):
                if cam_name == self.take_name_to_egocam_name[take_name]:
                    # 411 focal length. 1404/512 * 150 = 411
                    # originally, the images were 512x512 and we used a focal length of 150 to get most of the scene in the image
                    # but since our original source imageis now 1404, we need to scale the focal length by 1404/512
                    rectified_array, principal_points, focal_lengths = undistort_aria_given_device_calib(
                        self.take_name_to_egocam_fisheye_rgb_calibration[take_name], cv2_fisheye_rgb_data[local_frame_idx], "camera-rgb", self.rectified_ego_focal_length, full_image_size[0])
                elif cam_name in self.take_name_to_exocams_names[take_name]:
                    # 411 focal length. 1404/512 * 150 = 411
                    rectified_array, new_K_latest = undistort_exocam(cv2_fisheye_rgb_data[local_frame_idx],
                                                                    *self.take_name_to_exocams_calibration_parameters[take_name][cam_name], 
                                                                    (3840, 2160))
                else:
                    raise ValueError(f"Camera {cam_name} not in ego or exo cam list for {take_name}")
                
                # assert not (np.sum(rectified_array) == 0), "Rectified array is all zeros"
                rgb_frames[local_frame_idx] = torch.from_numpy(rectified_array)

            # assert not torch.sum(rgb_frames) == 0, f"Rectified array is all zeros for entire batch, must be bug. {sampled_frame_idxs_30fps.tolist()}"
        print(f"time taken to undistort video: {time.time() - t0}")

        t0 = time.time()
        if self.hand_detector is not None:
            # yolo_results = self.hand_detector([rgb_frames[local_frame_idx].data.cpu().numpy() for local_frame_idx in range(num_sampled_frames_for_this_video)], conf=0.3, verbose=False)
            # online_detections = build_detection_dict_from_yolo_results(yolo_results)
            hands23_results = self.hand_detector([rgb_frames[local_frame_idx].data.cpu().numpy() for local_frame_idx in range(num_sampled_frames_for_this_video)])
            online_detections = build_detection_dict_from_hands23_results(results_from_hands23(hands23_results))

        if self.early_return_rectified_frames:
            rgb_frames = rgb_frames[:num_sampled_frames_for_this_video]
            assert rgb_frames.shape[0] == sampled_frame_idxs_30fps.shape[0], f"Number of frames in rgb_frames and sampled_frame_idxs_30fps should match. {rgb_frames.shape[0]} != {sampled_frame_idxs_30fps.shape[0]}"
            assert rgb_frames.shape[0] > 0, "No frames found"
            return dict(rgb_frames=rgb_frames,
                        sampled_frame_idxs_30fps=sampled_frame_idxs_30fps,
                        take_name=[take_name] * num_sampled_frames_for_this_video,
                        cam_name=[cam_name] * num_sampled_frames_for_this_video)

        for local_idx, frame_idx in enumerate(sampled_frame_idxs_30fps.tolist()):
            if self.allow_takes_with_annotations_only and not self.frames_type_to_use == "all_frames":
                new_image_size = rgb_frames.shape[1]
                keypoint_dict = annotation_json[str(frame_idx)][0]['annotation2D'][cam_name]

                for key, value in keypoint_dict.items():
                    chilarity = key.split("_")[0]
                    hand_idx = 0 if chilarity == "left" else 1
                    joint_name = key.split(f"{chilarity}_")[1]

                    joint_idx = visualization_util.joint_string_to_idx(joint_name)
                    
                    # from old to new
                    old_to_new_scale = (new_image_size / full_image_size[0])

                    # we need to rotate the original annotation keypoints 90 degrees clockwise
                    # rotate first in the original image before scaling
                    cy = full_image_size[0] / 2
                    cx = full_image_size[1] / 2

                    if self.cam_type_to_use == "ego":
                        ninety_cw_transform  = torch.Tensor([[0, -1, cy + cx], [1, 0, -cx + cy], [0, 0, 1]])
                        annot_keypoint_in_original_image = (ninety_cw_transform @ torch.Tensor([value['x'], value['y'], 1]))
                        annot_keypoint_in_original_image = annot_keypoint_in_original_image[:2] / annot_keypoint_in_original_image[2]
                    else:
                        annot_keypoint_in_original_image = torch.Tensor([value['x'], value['y']])

                    # TODO: rn even for exo, we are calling it ego... but maybe this just works
                    ego_keypoint_2d_fullimg_tensor[local_idx, hand_idx, joint_idx, :] = annot_keypoint_in_original_image * old_to_new_scale
                    keypoint_present_tensor[local_idx, hand_idx, joint_idx] = True if value['placement'] == "manual" else False



            # put keypoints in keypoint_tensor
            # for missing keypoints, put in keypoint_present_tensor
            
            # debug viz
            if self.render_debug_frames:
                viz_img_gt = visualization_util.get_viz(
                    # to_pil_image(rgb_frames[local_idx]),
                    to_pil_image(rgb_frames[local_idx].permute(2, 0, 1)),
                    self.keypoints_map,
                    visualization_util.convert_2d_kp_to_dict(ego_keypoint_2d_fullimg_tensor[local_idx], keypoint_present_tensor[local_idx], self.keypoints_map),
                    self.skeleton,
                    self.pose_kpt_color,
                    annot_type="hands",
                    is_aria=False
                )
                rgb_debug_frames[local_idx] = torch.from_numpy(np.array(viz_img_gt)).to(torch.uint8)

            # rgb_debug_frames = np.array(rgb_debug_frames)
            
            # check that the bounding box presence matches the keypoint presence
            # we may have exo hands in here as well, so we need to pick out the ego hands
            # ego hands have centroids closest to the keypoint centroids
            # if self.bbox_json_root is not None:
            # bboxes_for_img = bbox_json_for_take[cam_name]['all_hands_found_bboxes_across_frames'][frame_idx]
            # ego_bboxes, ego_bboxes_present, ego_bboxes_confidences = get_ego_bounding_boxes_closest_to_keypoints(bboxes_for_img, ego_keypoint_2d_fullimg_tensor[local_idx], keypoint_present_tensor[local_idx])

            if self.bbox_npy_root is not None:
                if frame_idx in bbox_json_for_take[cam_name]['all_hands_found_boolean_across_frames'].keys():
                    ego_bboxes_present = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_boolean_across_frames'][frame_idx]).bool()
                    ego_bboxes = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_bboxes_across_frames'][frame_idx])
                    ego_bboxes_confidences = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_confidences_across_frames'][frame_idx])

                    # TODO: for 3D finetuning, we need to check that the 2D keypoints lie inside the box. if not, we say the bboxes is not present so we don't train on something wrong
                    # if we have a bbox and we have 2D keypoints, the 2D keypoints should lie inside the scaled box- otherwise 
                    # ego_hand_kp_2d = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_across_frames'][frame_idx])

                    if self.allow_takes_with_annotations_only:
                        for chilarity_tmp in range(2):
                            if not ego_bboxes_present[chilarity_tmp]:
                                continue
                            ego_bboxes_present[chilarity_tmp] = check_all_2d_kp_in_scaled_bbox(scale_bbox(ego_bboxes[chilarity_tmp], self.rescale_factor), ego_keypoint_2d_fullimg_tensor[local_idx, chilarity_tmp], 
                                                                                            keypoint_present_tensor[local_idx, chilarity_tmp])
                    ego_keypoint_2d_fullimg_tensor[local_idx] = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_across_frames'][frame_idx])
                else:
                    ego_bboxes_present = torch.zeros((2), dtype=torch.bool)
                    ego_bboxes = torch.zeros((2, 4), dtype=torch.float32)
                    ego_bboxes_confidences = torch.zeros((2), dtype=torch.float32)
                    # ego_hand_kp_2d = torch.zeros((2, 21, 2), dtype=torch.float32)  
            elif bool(self.bbox_ego_npy_filepath) or bool(self.bbox_exo_npy_filepath):
                if frame_idx in bbox_json_for_take[cam_name]['all_hands_found_boolean_across_frames'].keys():
                    ego_bboxes_present = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_boolean_across_frames'][frame_idx]).bool()
                    ego_bboxes = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_bboxes_across_frames'][frame_idx])
                    ego_bboxes_confidences = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_confidences_across_frames'][frame_idx])
                    ego_hand_kp_2d = torch.from_numpy(bbox_json_for_take[cam_name]['all_hands_found_across_frames'][frame_idx])
                else:
                    ego_bboxes_present = torch.zeros((2), dtype=torch.bool)
                    ego_bboxes = torch.zeros((2, 4), dtype=torch.float32)
                    ego_bboxes_confidences = torch.zeros((2), dtype=torch.float32)
                    ego_hand_kp_2d = torch.zeros((2, 21, 2), dtype=torch.float32)
                ego_keypoint_2d_fullimg_tensor[local_idx] = ego_hand_kp_2d
            else:
                # need local idx because online detections is indexed with batch-local indices
                bboxes_for_img = extract_single_bbox_data_from_dict(online_detections, local_idx)

                # -> 2, 4 and 2, 1
                ego_bboxes, ego_bboxes_present, ego_bboxes_confidences = get_ego_bounding_boxes_closest_to_keypoints(bboxes_for_img, ego_keypoint_2d_fullimg_tensor[local_idx], keypoint_present_tensor[local_idx])

            if self.right_hand_bbox_scaler is not None:
                if ego_bboxes_present[1]:
                    new_hw= self.right_hand_bbox_scaler.predict(make_bbox_features_from_x0y0x1y1(ego_bboxes[1:2]))
                    if np.any(new_hw <= 0):
                        print("ln533 hw prediction failed")
                    else:
                        ego_bboxes[1:2] = torch.from_numpy(make_x0y0x1y1_from_hwc(new_hw[:, 0], new_hw[:, 1], make_bbox_centers_from_x0y0x1y1(ego_bboxes[1:2]))).float()
            if self.left_hand_bbox_scaler is not None:
                if ego_bboxes_present[0]:
                    new_hw = self.left_hand_bbox_scaler.predict(make_bbox_features_from_x0y0x1y1(ego_bboxes[0:1]))
                    if np.any(new_hw <= 0):
                        print("ln540 hw prediction failed")
                    else:
                        ego_bboxes[0:1] = torch.from_numpy(make_x0y0x1y1_from_hwc(new_hw[:, 0], new_hw[:, 1], make_bbox_centers_from_x0y0x1y1(ego_bboxes[0:1]))).float()

            # get the image patch
            # -> 1, patch_size, patch_size, C
            # rgb_patches, keypoints_in_patch = get_image_batches_and_keypoints_from_bboxes(rgb_frames[local_idx], keypoint_2d_tensor[local_idx], ego_bboxes, ego_bboxes_present, self.rescale_factor, self.image_patch_size)
            
            # if the video is cached, then it likely has been downsized, and the original bboxes don't apply
            if not self.cached_rgb_dir:
                rgb_patches, affine_trans_arr = get_image_patches_from_bboxes(rgb_frames[local_idx], ego_keypoint_2d_fullimg_tensor[local_idx], ego_bboxes, ego_bboxes_present, self.rescale_factor, self.image_patch_size)
                rgb_patches_tensor[local_idx] = torch.from_numpy(rgb_patches)

                keypoints_in_patch =  get_keypoints_in_patch(ego_keypoint_2d_fullimg_tensor[local_idx], affine_trans_arr, ego_bboxes_present)

                ego_keypoint_2d_patches_tensor[local_idx] = torch.from_numpy(keypoints_in_patch).float()
            ego_bboxes_tensor[local_idx] = ego_bboxes
            ego_bboxes_present_tensor[local_idx] = ego_bboxes_present

            ego_bboxes_centers[local_idx] = (ego_bboxes[:, :2] + ego_bboxes[:, 2:]) / 2
            ego_bboxes_confidences_tensor[local_idx] = ego_bboxes_confidences


            bbox_scale = self.rescale_factor * (ego_bboxes[:, 2:4] - ego_bboxes[:, 0:2])

            # note: as seen later, the bounding box we use is always SQUARE
            bbox_size = bbox_scale.max(axis=-1)[0]

            ego_bboxes_sizes[local_idx] = bbox_size

            if not self.allow_takes_with_annotations_only:
                # if no annotations, the visibility is determined by the bounding box presence
                for chilarity_idx in range(2):
                    keypoint_present_tensor[local_idx, chilarity_idx, :] = ego_bboxes_present[chilarity_idx]
            

            # """
            # Patch debug block start
            # """
            # test_chilarity = 0
            # viz_img_gt = visualization_util.get_viz(
            #     to_pil_image(rgb_patches_tensor[local_idx][test_chilarity].permute(2, 0, 1)),
            #     self.keypoints_map,
            #     visualization_util.convert_2d_kp_to_dict(keypoint_patches_tensor[local_idx], keypoint_present_tensor[local_idx], self.keypoints_map, chilarity_to_show=test_chilarity),
            #     self.skeleton,
            #     self.pose_kpt_color,
            # )
            # viz_img_gt.save(f"06062025_patch_debug_{local_idx}.jpg")
            # """
            # Patch debug block end
            # """


        # """
        # Data for each sample
        #     sampled_frame_idxs_30fps: inflate_length
        #     take_name: str
        #     rgb_frame: inflate_length, H, W, C X
        #     rgb_debug_frames: inflate_length, H, W, C X
        #     rgb_patches: inflate_length, 2,patch_size, patch_size, C, X. NOTE: these patches are right hand only because they have been flipped.
        #     bbox:
        #     keypoints_2d_tensor: inflate_length, 2, 21, 2 X.
        #     keypoint_present_tensor (per joint visibility): inflate_length, 2, 21
        #     ego_bboxes_tensor: inflate_length, 2, 4 X
        #     ego_bboxes_present_tensor: inflate_length, 2, 1 X
        #     keypoint_patch_tensor: inflate_length, 2, 21, 2 X. NOTE: these patches are right hand only because they have been flipped.
        #     ego_bboxes_centers: inflate_length, 2, 2
        #     ego_bboxes_sizes: inflate_length, 2
        # """

        """
        Build aria intrinsic manually
        """
        aria_intrinsic_matrix = np.eye(3)
        aria_intrinsic_matrix[0, 0] = self.rectified_ego_focal_length
        aria_intrinsic_matrix[1, 1] = self.rectified_ego_focal_length
        aria_intrinsic_matrix[0, 2] = 1404 / 2 # TODO: for now, just return it for the full sized image...
        aria_intrinsic_matrix[1, 2] = 1404 / 2

        # make the bbox false if all joints are missing
        if self.allow_takes_with_annotations_only:
            all_joints_gone = torch.all(keypoint_present_tensor == False, dim=-1)
            ego_bboxes_present_tensor[all_joints_gone] = False

        # add colorjitter augmentation to rgb patches tensor
        if self.colorjitter_augmentation:
            rgb_patches_tensor = colorjitter_augmentation(rgb_patches_tensor)

        # pad sampled_frame_idxs_30fps to num_placeholder_frames 
        sampled_frame_idxs_30fps = torch.cat([sampled_frame_idxs_30fps, torch.zeros(num_placeholder_frames - sampled_frame_idxs_30fps.shape[0], dtype=torch.int32)], dim=0)
        
        out_dict = dict(
            sampled_frame_idxs_30fps=sampled_frame_idxs_30fps,
            take_name=[take_name] * num_sampled_frames_for_this_video,
            cam_name=[cam_name] * num_sampled_frames_for_this_video,
            rgb_frames=rgb_frames if self.return_full_images else dict(),
            rgb_cached_frames=rgb_cached_frames if self.cached_rgb_dir else dict(),
            rgb_debug_frames=rgb_debug_frames if self.return_full_images else dict(),
            ego_keypoint_2d_fullimg_tensor=ego_keypoint_2d_fullimg_tensor,
            keypoint_present_tensor=keypoint_present_tensor,
            ego_bboxes_tensor=ego_bboxes_tensor,
            ego_bboxes_present_tensor=ego_bboxes_present_tensor,
            rgb_patches_tensor=rgb_patches_tensor,
            ego_keypoint_2d_patches_tensor=ego_keypoint_2d_patches_tensor, # this is keypoints_2d_patch. The ground truth keypoints in the RIGHT HAND patch.
            ego_bboxes_centers=ego_bboxes_centers,
            ego_bboxes_sizes=ego_bboxes_sizes,
            ego_bboxes_confidences_tensor=ego_bboxes_confidences_tensor,
            exo_cam_wrt_world={cam_name: np.load(self.take_name_to_exo_cam_wrt_world_save_path[take_name][cam_name]) for cam_name in self.take_name_to_exocams_names[take_name]} if self.load_cam_data else {},
            exo_cam_intrinsics={cam_name: np.load(self.take_name_to_exo_cam_intrinsics_save_path[take_name][cam_name]) for cam_name in self.take_name_to_exocams_names[take_name]} if self.load_cam_data else {},
            aria_cam_wrt_world=np.load(self.take_name_to_aria_cam_wrt_world_save_path[take_name])[sampled_frame_idxs_30fps] if self.load_cam_data else {}, # this line may fail if you are launching many jobs and they keep rewriting each other
            aria_cam_intrinsics=aria_intrinsic_matrix[np.newaxis, :, :].repeat(num_sampled_frames_for_this_video, axis=0) if self.load_cam_data else {},
            last_partition_boolean=self.sample_idx_to_last_partition_boolean[sample_idx] if self.frames_type_to_use == "iterate_all_frames" else {},
            dataset_sample_idx=sample_idx,
            ego_keypoint_3d_wrtworld_tensor=ego_keypoint_3d_wrtworld_tensor,
            ego_keypoint_3d_present_tensor=ego_keypoint_3d_present_tensor
        )
        print(f"time taken to getitem (rest of function): {time.time() - t0}")
        print(f"absolute time {time.time()} \n\n")
        return out_dict

if __name__ == "__main__":
    from line_profiler import LineProfiler

    torch.manual_seed(0)
    np.random.seed(0)
    # unit test
    # dataset = EgoExoDatasetFromGTJson(
    #     # image_root="/data/pulkitag/hamer_diffusion_policy_project/scratch/05292025_iiitcooking1342_randomsampled_200_50",
    #     # release_dir="/data/pulkitag/models/rli14/data/egoexo",
    #     release_dir="/data/pulkitag/hamer_diffusion_policy_project/datasets/egoexo/",
    #     annotation_root="/data/pulkitag/models/rli14/data/egoexo/annotations/ego_pose/train/hand/annotation",
    #     # bbox_json_path="/data/scratch-oc40/pulkitag/rli14/hamer_diffusion_policy/labels/05242025_iiitcooking1342/iiith_cooking_134_2/bbox.json",
    #     bbox_json_root="/data/pulkitag/hamer_diffusion_policy_project/labels/06062025_egoexo_gtannotation_image_export",
    #     inflate_length=10,
    #     render_debug_frames=False,
    #     iterate_over_frames=False,
    #     debug_max_takes=1e10,
    #     allowed_parent_tasks=["Cooking"],
    #     allowed_task_names=["Cooking Tomato & Eggs"],
    #     frames_type_to_use="all_frames"
    # )

    dataset = EgoExoDatasetFromGTJson(
        release_dir="/data/pulkitag/hamer_diffusion_policy_project/datasets/egoexo/",
        annotation_root="/data/pulkitag/models/rli14/data/egoexo/annotations/ego_pose/train/hand/annotation",
        bbox_npy_root="/data/pulkitag/hamer_diffusion_policy_project/datasets/egoexo_generated/0630_egg_and_tomato_full",
        num_subclips_per_video=100, # clips per batch / video
        render_debug_frames=False,
        debug_max_takes=10,
        return_full_images=False,
        allow_takes_with_annotations_only=False,
        # allowed_take_names=["iiith_cooking_134_2"],
        frames_type_to_use="valid_3d_kp_clips",
        hand_detector=None,
        partition_size=4, 
        early_return_rectified_frames=False,
        allowed_parent_tasks=["Cooking"],
        allowed_task_names=["Cooking Tomato & Eggs"],
        cam_type_to_use="ego",
        right_hand_bbox_scaler=None,
        left_hand_bbox_scaler=None,
        colorjitter_augmentation=True,
        three_d_keypoints_torch_root="/data/pulkitag/hamer_diffusion_policy_project/datasets/egoexo_generated/0708_egg_and_tomato_full_triangulate_retry_reproj14_44_noviz/",
        cached_rgb_dir="/data/pulkitag/hamer_diffusion_policy_project/datasets/egoexo_generated/0705_egg_and_tomato_full_rectified_export/",
        filter_clip_proprio_norm=.05,
        joint_wrt_cam_cache_dir="/data/pulkitag/hamer_diffusion_policy_project/projects/WiLoR-mini/scratch/dataset_cache"
    )

    profiler = LineProfiler()
    profiler.add_function(EgoExoDatasetFromGTJson.__getitem__)

    profiler.enable()

    # sample = dataset[9]
    # sample = dataset[0]

    DEBUG_MAX_TAKES = 4
    for i in range(DEBUG_MAX_TAKES):
        dataset.__getitem__(i)
    profiler.disable()
    profiler.print_stats()

    import pdb; pdb.set_trace()
    # """
    # Below block lets us iterate over the frames of the dataset
    # """
    # worker_init_fn = lambda x: np.random.seed(x + 1)

    # dataloader = DataLoader(dataset, batch_size=1, num_workers=1, worker_init_fn=worker_init_fn, pin_memory=True,
    #                         shuffle=False, return_rectified_frames=True)

    # out_dir = "/data/pulkitag/hamer_diffusion_policy_project/scratch/06062025_egoexo_gtannotation_image_export"
    # debug_out_dir = "/data/pulkitag/hamer_diffusion_policy_project/scratch/06062025_egoexo_gtannotation_image_export_debug"
    # for batch_idx, (sampled_frame_idxs_30fps, take_name_list, rgb_frames, rgb_debug_frames) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    #     # print(sampled_frame_idxs_30fps)
    #     # print(rgb_frames.shape)
    #     # print(len(take_name_list))

    #     os.makedirs(os.path.join(out_dir, take_name_list[0]), exist_ok=True)
    #     os.makedirs(os.path.join(debug_out_dir, take_name_list[0]), exist_ok=True)
    #     # because the batch_size is 1, we can just take the first element
    #     for local_frame_idx, (frame_idx, rgb_frame) in enumerate(zip(sampled_frame_idxs_30fps[0], rgb_frames[0])):
    #         cv2.imwrite(os.path.join(out_dir, take_name_list[0], f"{frame_idx:06d}.jpg"), rgb_frame.data.cpu().numpy().astype(np.uint8)[..., ::-1])

    #         cv2.imwrite(os.path.join(debug_out_dir, take_name_list[0], f"{frame_idx:06d}.jpg"), rgb_debug_frames[0][local_frame_idx].data.cpu().numpy().astype(np.uint8)[..., ::-1])

    print("Done!")
    # profiler.disable()
    # profiler.print_stats()