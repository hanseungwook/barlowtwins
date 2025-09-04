rgb_cached_frames: 224x224 processed images ready for training
sampled_frame_idxs_30fps: first frames of the subclip. the rgb frames land on this timestep
ego_keypoint_3d_wrtworld_tensor: actions in world frame. not suitable for training
egoexo_nproprio: proprio wrt to cam frame.
egoexo_nactions: action wrt to cam frame.

sample_clips_with_text_annotations_only: True samples language annotations

text_labels: the text labels, one per sample / batch (batch size 1)

take_name: the take names, one per sample / batch. same size as text_labels