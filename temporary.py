import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_epi_len, action_hack=False):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_epi_len = max_epi_len
        self.action_hack = action_hack

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            action_shape = root['/action'].shape
            episode_len = action_shape[0]
            action_dim = action_shape[1]
            start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts]  # (14,)

            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/{cam_name}'][start_ts]  # (H,W,3)

            if not self.action_hack:
                action = root['/action'][start_ts:]   # (T', action_dim)
                action_len = episode_len - start_ts
            else:
                action_start = max(0, start_ts - 1)
                action = root['/action'][action_start:]
                action_len = episode_len - action_start

            # Current observation modalities
            env_pcd = root['/observations/env_pcd'][start_ts]          # (2048, 3)
            garment_pcd = root['/observations/garment_pcd'][()]       # (2048, 3), episode-level fixed
            stage = root['/stage'][start_ts]                          # scalar, current stage label
            previous_stage = root['/previous_stage'][start_ts]        # scalar, input stage index
            affordance_score = root['/observations/affordance_score'][()]  # (4, 2048)

        # Pad action
        padded_action = np.zeros((self.max_epi_len, action_dim), dtype=np.float32)
        is_pad = np.ones(self.max_epi_len, dtype=np.bool_)
        padded_action[:action_len] = action
        is_pad[:action_len] = False

        # Stack images
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)  # (K, H, W, C)

        # Normalize point clouds using shared env stats
        env_pcd = normalize_pcd_with_env_stats(
            env_pcd,
            self.norm_stats["env_pcd_min"],
            self.norm_stats["env_pcd_max"],
        )  # (2048, 3)

        garment_pcd = normalize_pcd_with_env_stats(
            garment_pcd,
            self.norm_stats["env_pcd_min"],
            self.norm_stats["env_pcd_max"],
        )  # (2048, 3)

        # To torch
        image_data = torch.from_numpy(all_cam_images)                 # (K,H,W,C)
        qpos_data = torch.from_numpy(qpos).float()                    # (14,)
        action_data = torch.from_numpy(padded_action).float()         # (Tmax, action_dim)
        is_pad = torch.from_numpy(is_pad)                             # (Tmax,)

        env_pcd_data = torch.from_numpy(env_pcd).float()              # (2048, 3)
        garment_pcd_data = torch.from_numpy(garment_pcd).float()      # (2048, 3)
        stage_data = torch.tensor(stage, dtype=torch.long)            # ()
        previous_stage_data = torch.tensor(previous_stage, dtype=torch.long)  # ()
        affordance_score_data = torch.from_numpy(affordance_score).float()    # (4, 2048)

        # Image: channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)   # (K,C,H,W)
        image_data = image_data.float() / 255.0

        # Mean-std normalization
        action_data = (action_data - torch.from_numpy(self.norm_stats["action_mean"])) / \
                      torch.from_numpy(self.norm_stats["action_std"])

        qpos_data = (qpos_data - torch.from_numpy(self.norm_stats["qpos_mean"])) / \
                    torch.from_numpy(self.norm_stats["qpos_std"])

        return (
            image_data,
            qpos_data,
            action_data,
            is_pad,
            env_pcd_data,
            garment_pcd_data,
            stage_data,
            previous_stage_data,
            affordance_score_data,
        )
    
def normalize_pcd_with_env_stats(pcd, env_pcd_min, env_pcd_max):
    """
    pcd: numpy array, shape (N, 3)
    env_pcd_min: numpy array, shape (3,)
    env_pcd_max: numpy array, shape (3,)
    """
    pcd = (pcd - env_pcd_min) / (env_pcd_max - env_pcd_min)
    pcd = np.clip(pcd, 0.0, 1.0)
    return pcd.astype(np.float32)













def weighted_pool(feat, weight, eps=1e-6):
    """
    feat:   (B, N, D)
    weight: (B, N)
    return: (B, D)
    """
    w = weight.unsqueeze(-1)  # (B, N, 1)
    pooled = (feat * w).sum(dim=1) / (w.sum(dim=1) + eps)
    return pooled


def build_garment_tokens(garment_feat, affordance_score_all, previous_stage, null_token):
    """
    garment_feat:         (B, 2048, D)
    affordance_score_all: (B, 4, 2048)
    previous_stage:       (B,)
    null_token:           (D,)

    output:
        garment_tokens:   (B, 2, D)
    """
    B, N, D = garment_feat.shape
    device = garment_feat.device

    # Precompute all atomic part tokens
    left_sleeve_tok = weighted_pool(garment_feat, affordance_score_all[:, 0, :])   # (B, D)
    right_sleeve_tok = weighted_pool(garment_feat, affordance_score_all[:, 1, :])  # (B, D)
    left_bottom_tok = weighted_pool(garment_feat, affordance_score_all[:, 2, :])   # (B, D)
    right_bottom_tok = weighted_pool(garment_feat, affordance_score_all[:, 3, :])  # (B, D)

    null_tok = null_token.unsqueeze(0).expand(B, -1)  # (B, D)

    token1 = torch.zeros(B, D, device=device)
    token2 = torch.zeros(B, D, device=device)

    # stage 0 -> [right sleeve, null]
    mask0 = (previous_stage == 0)
    token1[mask0] = right_sleeve_tok[mask0]
    token2[mask0] = null_tok[mask0]

    # stage 1 -> [left sleeve, null]
    mask1 = (previous_stage == 1)
    token1[mask1] = left_sleeve_tok[mask1]
    token2[mask1] = null_tok[mask1]

    # stage 2 -> [left bottom, right bottom]
    mask2 = (previous_stage == 2)
    token1[mask2] = left_bottom_tok[mask2]
    token2[mask2] = right_bottom_tok[mask2]

    garment_tokens = torch.stack([token1, token2], dim=1)  # (B, 2, D)
    return garment_tokens