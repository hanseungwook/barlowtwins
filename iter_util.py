import numpy as np
import torch

class CycleIter:
    """
    Allows endlessly calling next on a dataloader
    """
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)   # restart
            return next(self.it)

def iter_batches(mode, egoexo_dataloader, lerobot_train_dataloader, cfg):
    if mode == "egoexo":
        for egoexo_batch in iter(egoexo_dataloader):
            robot_batch = None

            yield {"egoexo": egoexo_batch, "robot": robot_batch}
    elif mode == "robot":
        if cfg.training.load_lerobot_into_memory:
            for robot_batch_idx in range(len(lerobot_train_dataloader)):
                ep_indices = np.random.choice(np.arange(lerobot_train_dataset.num_episodes), size=lerobot_batch_size, replace=True)

                # from: inclusive
                # to: exclusive
                lerobot_sampled_indices = np.random.randint(lerobot_train_dataset.episode_data_index["from"][ep_indices], 
                lerobot_train_dataset.episode_data_index["to"][ep_indices]-cfg.horizon)

                robot_batch = {
                    "observation.image.ego_global": torch.from_numpy(lerobot_observation_image_ego_global[lerobot_sampled_indices]), # B, H, W, C
                    "action": torch.from_numpy(lerobot_action[lerobot_sampled_indices[:, np.newaxis] + np.arange(cfg.horizon)[np.newaxis, :]]), # B, H, 10
                    "observation.state": torch.from_numpy(lerobot_observation_state[lerobot_sampled_indices[:, np.newaxis] + np.arange(cfg.n_obs_steps)[np.newaxis, :]]) # B, H, 10
                }
                yield {"egoexo": None, "robot": robot_batch}
        else:
            for robot_batch in iter(lerobot_train_dataloader):
                yield {"egoexo": None, "robot": robot_batch}
    elif mode == "both":
        max_len = max(len(egoexo_dataloader), len(lerobot_train_dataloader))

        cycle_egoexo_dataloader = CycleIter(egoexo_dataloader)
        cycle_lerobot_train_dataloader = CycleIter(lerobot_train_dataloader)

        for batch_idx in range(max_len):
            # TODO: add timing here
            egoexo_batch = next(cycle_egoexo_dataloader)
            robot_batch = next(cycle_lerobot_train_dataloader)

            yield {"egoexo": egoexo_batch, "robot": robot_batch}
    else:
        raise ValueError(f"Invalid mode: {mode}")
