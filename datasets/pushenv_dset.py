import torch

import pickle
import numpy as np
from pathlib import Path
from einops import rearrange

from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
import cv2  # new dependency


# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([282.9706, 363.5293, 275.0370, 383.0857, -0.6950])
STATE_STD = torch.tensor([103.7191, 106.0112, 85.7054, 91.4971, 2.5156])
PROPRIO_MEAN = torch.tensor([282.9706, 363.5293])
PROPRIO_STD = torch.tensor([103.7191, 106.0112])

class PushEnvDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pushenv_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['X_img'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)


        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pushenv, last 2 dim of states is proprio
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    # def get_frames(self, idx, frames):
    #     vid_dir = self.data_path / "obses"
    #     reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
    #     act = self.actions[idx, frames]
    #     state = self.states[idx, frames]
    #     proprio = self.proprios[idx, frames]
    #     shape = self.shapes[idx]

    #     image = reader.get_batch(frames)  # THWC
    #     image = image / 255.0
    #     image = rearrange(image, "T H W C -> T C H W")
    #     if self.transform:
    #         image = self.transform(image)
    #     obs = {"visual": image, "proprio": proprio}
    #     return obs, act, state, {'shape': shape}

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        video_path = str(vid_dir / f"episode_{idx:03d}.mp4")
        
        cap = cv2.VideoCapture(video_path)
        images = []
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, frame_img = cap.read()
            if not ret:
                raise ValueError(f"Frame {frame} could not be read from {video_path}")
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            images.append(frame_img)
        cap.release()
        image_array = np.stack(images, axis=0) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(0, 3, 1, 2)

        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]
        
        obs = {"visual": image_tensor, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pushenv_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,

):
    train_dset = PushEnvDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
    )
    val_dset = PushEnvDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset