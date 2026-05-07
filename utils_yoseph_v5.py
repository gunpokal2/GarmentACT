import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    ### yoseph ###
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_epi_len, action_hack=False):
    ### yoseph ###
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids # (40) / (10)
        self.dataset_dir = dataset_dir # '/data2/garment/yoseph/dataset'
        self.camera_names = camera_names # ['front', 'left', 'right']
        self.norm_stats = norm_stats 
        ### yoseph ###
        # action, qpos의 경우 mean-std normalization이 더 좋다
        # garment_pcd, env_pcd의 경우 env_pcd의 min-max로 normalization...
        # norm_stats['action_mean'] (14), norm_stats['action_std'] (14), norm_stats['qpos_mean'] (14), norm_stats['qpos_std'] (14), norm_stats['env_pcd_max'] (3), norm_stats['env_pcd_min'] (3)
        self.max_epi_len = max_epi_len
        self.action_hack = action_hack

        # self.is_sim = None
        # self.__getitem__(0) # initialize self.is_sim

        self.action_mean = torch.from_numpy(self.norm_stats["action_mean"]).float()
        self.action_std = torch.from_numpy(self.norm_stats["action_std"]).float()
        self.qpos_mean = torch.from_numpy(self.norm_stats["qpos_mean"]).float()
        self.qpos_std = torch.from_numpy(self.norm_stats["qpos_std"]).float()
        self.env_pcd_min = torch.from_numpy(self.norm_stats["env_pcd_min"]).float()
        self.env_pcd_max = torch.from_numpy(self.norm_stats["env_pcd_max"]).float()
        ### yoseph ###

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

            qpos = root['/observations/qpos'][start_ts]

            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/{cam_name}'][start_ts] # (480, 640, 3)

            if not self.action_hack:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action_start = max(0, start_ts - 1)
                action = root['/action'][action_start:]
                action_len = episode_len - action_start
            
            ### yoseph ###
            # env_pcd = root['/observations/env_pcd'][start_ts] # (2048, 3)
            # garment_pcd = root['/observations/garment_pcd'][:] # (2048, 3), episode-level fixed
            # stage = root['/stage'][start_ts] # scalar, current stage label
            # previous_stage = root['/previous_stage'][start_ts] # scalar, input stage index
            # affordance_score = root['/affordance_score'][:] # (4, 2048), episode-level fixed
            ### yoseph ###

        padded_action = np.zeros((self.max_epi_len, action_dim), dtype=np.float32)
        is_pad = np.ones(self.max_epi_len, dtype=np.bool_)
        padded_action[:action_len] = action
        is_pad[:action_len] = False

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0) # (3, 480, 640, 3)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad)
        ### yoseph ###
        # env_pcd_data = torch.from_numpy(env_pcd).float()              # (2048, 3)
        # garment_pcd_data = torch.from_numpy(garment_pcd).float()      # (2048, 3)
        # stage_data = torch.tensor(stage, dtype=torch.long)            # ()
        # previous_stage_data = torch.tensor(previous_stage, dtype=torch.long)  # ()
        # affordance_score_data = torch.from_numpy(affordance_score).float()    # (4, 2048)
        ### yoseph ###

        image_data = torch.einsum('k h w c -> k c h w', image_data) # (3, 3, 480, 640)
        image_data = image_data.float() / 255.0

        action_data = (action_data - self.action_mean) / self.action_std
        qpos_data = (qpos_data - self.qpos_mean) / self.qpos_std

        # env_pcd_data = (env_pcd_data - self.env_pcd_min) / (self.env_pcd_max - self.env_pcd_min)
        # env_pcd_data = torch.clamp(env_pcd_data, 0.0, 1.0)

        # garment_pcd_data = (garment_pcd_data - self.env_pcd_min) / (self.env_pcd_max - self.env_pcd_min)
        # garment_pcd_data = torch.clamp(garment_pcd_data, 0.0, 1.0)
        
        # return image_data, qpos_data, action_data, is_pad, env_pcd_data, garment_pcd_data, stage_data, previous_stage_data, affordance_score_data
        # return image_data, qpos_data, action_data, is_pad, env_pcd_data
        return image_data, qpos_data, action_data, is_pad


def get_max_epi_len(dataset_dir, num_episodes):
    max_len = 0
    for episode_id in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            ep_len = root['/action'].shape[0]
            max_len = max(max_len, ep_len)
    return max_len

def get_norm_stats(dataset_dir, num_episodes):
    ### must be changed ###
    # ENV_PCD_MIN = np.array([-0.54, -0.27, 0.002], dtype=np.float32)
    # ENV_PCD_MAX = np.array([0.54, 0.27, 0.486], dtype=np.float32)
    ENV_PCD_MIN = np.array([-0.54, -0.27, 0.002], dtype=np.float32)
    ENV_PCD_MAX = np.array([0.51, 0.29, 0.475], dtype=np.float32)
    ### must be changed ###
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]     # (T_i, 14)
            action = root['/action'][()]              # (T_i, 14)

        all_qpos_data.append(qpos)
        all_action_data.append(action)

    all_qpos_data = np.concatenate(all_qpos_data, axis=0)     # (sum_i T_i, 14)
    all_action_data = np.concatenate(all_action_data, axis=0) # (sum_i T_i, 14)

    action_mean = all_action_data.mean(axis=0).astype(np.float32)
    action_std = all_action_data.std(axis=0).astype(np.float32)
    action_std = np.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(axis=0).astype(np.float32)
    qpos_std = all_qpos_data.std(axis=0).astype(np.float32)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean,   # (14,)
        "action_std": action_std,     # (14,)
        "qpos_mean": qpos_mean,       # (14,)
        "qpos_std": qpos_std,         # (14,)
        "env_pcd_min": ENV_PCD_MIN,   # (3,)
        "env_pcd_max": ENV_PCD_MAX,   # (3,)
    }
    return stats

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    ### yoseph ###
    max_epi_len = get_max_epi_len(dataset_dir, num_episodes)

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_epi_len)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_epi_len)
    ### yoseph ###
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    ### yoseph ###
    return train_dataloader, val_dataloader, norm_stats
    ### yoseph ###

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# class EpisodicDataset(torch.utils.data.Dataset):
#     def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
#         super(EpisodicDataset).__init__()
#         self.episode_ids = episode_ids # (40) / (10)
#         self.dataset_dir = dataset_dir # '/data2/act/dataset/sim_transfer_cube_scripted'
#         self.camera_names = camera_names # ['top']
#         self.norm_stats = norm_stats # norm_stats['action_mean'] (14), norm_stats['action_std'] (14), norm_stats['qpos_mean'] (14), norm_stats['qpos_std'] (14), norm_stats['example_qpos'] (400, 14)
#         self.is_sim = None
#         self.__getitem__(0) # initialize self.is_sim

#     def __len__(self):
#         return len(self.episode_ids)

#     def __getitem__(self, index):
#         sample_full_episode = False # hardcode

#         episode_id = self.episode_ids[index]
#         dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             is_sim = root.attrs['sim']
#             original_action_shape = root['/action'].shape # (400, 14)
#             episode_len = original_action_shape[0] # 400
#             if sample_full_episode: # False
#                 start_ts = 0
#             else:
#                 start_ts = np.random.choice(episode_len) # 265
#             # get observation at start_ts only
#             qpos = root['/observations/qpos'][start_ts] # (14)
#             qvel = root['/observations/qvel'][start_ts]
#             image_dict = dict()
#             for cam_name in self.camera_names:
#                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts] # image_dict['top'] (480, 640, 3)
#             # get all actions after and including start_ts 
#             # simulator에서는 보통 log가 깔끔해서 obs[t]와 대응되는 action sequence를 action[t:]로 써도 정렬이 잘 맞는다... 
#             # real robot에서는 log가 1 step씩 어긋나는 경우가 흔해서 action[max(0, t-1):]로 한 칸 당겨 더 잘 맞는 alignment를 만들도록 hacking...!
#             if is_sim:
#                 action = root['/action'][start_ts:] # (135, 14)
#                 action_len = episode_len - start_ts # 135
#             else:
#                 action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
#                 action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
#         # 하는 일 1: action을 zero-padding을 해버리면서 동시에 is_pad mask를 만든다...
#         self.is_sim = is_sim
#         padded_action = np.zeros(original_action_shape, dtype=np.float32) # (400, 14)
#         padded_action[:action_len] = action
#         is_pad = np.zeros(episode_len)
#         is_pad[action_len:] = 1

#         # new axis for different cameras
#         all_cam_images = []
#         for cam_name in self.camera_names:
#             all_cam_images.append(image_dict[cam_name])
#         all_cam_images = np.stack(all_cam_images, axis=0) # (1, 480, 640, 3)

#         # construct observations
#         image_data = torch.from_numpy(all_cam_images)
#         qpos_data = torch.from_numpy(qpos).float() # (14)
#         action_data = torch.from_numpy(padded_action).float() # (400, 14)
#         is_pad = torch.from_numpy(is_pad).bool() # (400)

#         # channel last
#         image_data = torch.einsum('k h w c -> k c h w', image_data) # (1, 3, 480, 640)
#         # 하는 일 2: image, action, qpos normalize
#         # normalize image and change dtype to float
#         image_data = image_data / 255.0
#         action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
#         qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

#         return image_data, qpos_data, action_data, is_pad


# def get_norm_stats(dataset_dir, num_episodes):
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in range(num_episodes): # 50
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()] # (400, 14)
#             qvel = root['/observations/qvel'][()] # (400, 14)
#             action = root['/action'][()] # (400, 14)
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))
#     all_qpos_data = torch.stack(all_qpos_data) # (50, 400, 14)
#     all_action_data = torch.stack(all_action_data) # (50, 400, 14)
#     all_action_data = all_action_data
#     # epi_num = 50, epi_len = 400
#     # normalize action data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True) # (1, 1, 14)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True) # (1, 1, 14)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     # normalize qpos data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True) # (1, 1, 14)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True) # (1, 1, 14)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
#              "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
#              "example_qpos": qpos}

#     return stats

# def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
#     print(f'\nData from: {dataset_dir}\n')
#     # obtain train test split
#     train_ratio = 0.8
#     shuffled_indices = np.random.permutation(num_episodes) # (50), array([27, 35, 40, 38, 2, ..., 37])
#     train_indices = shuffled_indices[:int(train_ratio * num_episodes)] # (40)
#     val_indices = shuffled_indices[int(train_ratio * num_episodes):] # (10)

#     # obtain normalization stats for qpos and action
#     norm_stats = get_norm_stats(dataset_dir, num_episodes)
#     # norm_stats['action_mean'] (14), norm_stats['action_std'] (14), norm_stats['qpos_mean'] (14), norm_stats['qpos_std'] (14), norm_stats['example_qpos'] (400, 14)
#     # construct dataset and dataloader
#     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
#     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1) # batch_size_train = 8
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1) # batch_size_val = 8

#     return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# ### env utils

# def sample_box_pose():
#     x_range = [0.0, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     cube_quat = np.array([1, 0, 0, 0])
#     return np.concatenate([cube_position, cube_quat])

# def sample_insertion_pose():
#     # Peg
#     x_range = [0.1, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     peg_quat = np.array([1, 0, 0, 0])
#     peg_pose = np.concatenate([peg_position, peg_quat])

#     # Socket
#     x_range = [-0.2, -0.1]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     socket_quat = np.array([1, 0, 0, 0])
#     socket_pose = np.concatenate([socket_position, socket_quat])

#     return peg_pose, socket_pose

# ### helper functions

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result

# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
