import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


### encoder candidates ###
class PointNet2(nn.Module):
    
    def __init__(self, normal_channel=False, feature_dim=128):
        super(PointNet2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, feature_dim, 1)  # 输出 feature_dim 维特征向量

    def forward(self, xyz): # (B, 2048, 3)
        # Set Abstraction layers
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz # (B, 3, 2048)
            l0_xyz = xyz # (B, 3, 2048)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 512), (B, 128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128), (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 1), (B, 1024, 1)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 512)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points) # (B, 128, 2048)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points))) # (B, 128, 2048)
        output = self.drop1(feat)
        output = self.conv2(output) # (B, 128, 2048)
        output = output.permute(0, 2, 1) # (B, 2048, 128) # 输出形状 (B, N, feature_dim)
        return output


class PointNet2Global(nn.Module):
    def __init__(self, affordance_feature=False, feature_dim=32):
        super(PointNet2Global, self).__init__()
        if affordance_feature: # True
            additional_channel = 2
        else:
            additional_channel = 0
        self.affordance_feature = affordance_feature # True
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim), # 64
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
    def forward(self, xyz):
        # Set Abstraction layers
        if self.affordance_feature: # True
            if xyz.shape[1] != 5: # True
                xyz = xyz.permute(0, 2, 1) # (B*3, 5, 2048)
        else:
            if xyz.shape[1] != 3:
                xyz = xyz.permute(0, 2, 1)
                
        B,C,N = xyz.shape
        
        if self.affordance_feature: # True
            l0_points = xyz # (B*3, 5, 2048)
            l0_xyz = xyz[:,:3,:] # (B*3, 3, 2048)
        else:
            l0_points = xyz
            l0_xyz = xyz
       
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # max_pooling
        global_feat = l3_points.squeeze(-1)  # (B, 512)
        # fully connected layers, downsample to feature_dim
        global_feat = self.fc(global_feat)
        return global_feat
    
class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(
            f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red"
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection # True
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x): # (B*3, 2048, 3)
        x = self.mlp(x) # (B*3, 2048, 256)
        x = torch.max(x, 1)[0] # (B*3, 256) 여기서 그냥 max pooling 때려버리는 건 너무... 단순무식하지 않나? env pointcloud라 그냥 대충 해버리나...
        x = self.final_projection(x) # (B*3, 128) 
        return x
### encoder candidates ###

# self.extractor_garment = PointNet2Global(
#     affordance_feature=True,
#     feature_dim=out_channel//2, # 64
# )


# self.extractor_env = PointNetEncoderXYZ(
#     in_channels=3,
#     out_channels=out_channel, # 128
#     use_layernorm=True,
#     final_norm="layernorm",
# )

def build_pointnet2():
    garment_pcd_encoder = PointNet2(feature_dim=128, normal_channel=False) # 바꿔줘야할 듯

    env_pcd_encoder = PointNetEncoderXYZ(
        in_channels=3,
        out_channels=128, # 바꿔줘야할 듯
        use_layernorm=True,
        final_norm="layernorm",
    )

    return garment_pcd_encoder, env_pcd_encoder



def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    garment_encoder = PointNet2(feature_dim=128, normal_channel=False).to(device)
    points = np.random.rand(2, 2048, 3)    
    print(points.shape)    
    output = garment_encoder(torch.from_numpy(points).float().to(device))
    print(output.shape)



if __name__ == '__main__':
    main()