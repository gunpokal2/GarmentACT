import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst): # src (B,N,C), dst (B,M,C)가 입력으로 주어지면 dist^2 (B,N,M)을 출력으로
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx): 
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) # [B, S] / [B, S, K]
    view_shape[1:] = [1] * (len(view_shape) - 1) # [B, 1] / [B, 1, 1]
    repeat_shape = list(idx.shape) # [B, S] / [B, S, K]
    repeat_shape[0] = 1 # [1, S] / [1, S, K]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # (B,S) 여기서 각 행은 batch index가 S번 반복된 것 / (B,S,K)
    new_points = points[batch_indices, idx, :]
    return new_points # (B,S,C) / (B,S,K,C)


def farthest_point_sample(xyz, npoint): # xyz (B, N, 3)가 입력으로 주어지면 centroids (B, S=npoint)가 출력으로 주어진다... 출력은 좌표가 아니라 index들
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # centroids[b, i]는 b번째 batch에서 i번째로 선택된 대표점의 index
    distance = torch.ones(B, N).to(device) * 1e10 # distance[b, i]는 점 i가 현재까지 선택된 대표점들 중 가장 가까운 것까지의 거리^2. 처음엔 대표점이 없으니 매우 큰 값(1e10)으로 시작
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # farthest[b]는 다음에 뽑을 대표점 index (batch마다 하나씩) 첫 시작은 random 한개
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest # 모든 batch에서 i번째로 선택된 대표점의 index
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) # (B, 1, 3) 방금 뽑은 대표점 좌표 가져오기
        dist = torch.sum((xyz - centroid) ** 2, -1) # (B, N) 모든 점과의 거리^2 계산
        mask = dist < distance # (B, N) 최소 거리가 유지되도록 distance update
        distance[mask] = dist[mask] # (B, N) 새 대표점을 추가했으니 새 대표점이 더 가까우면 그 최소 거리 값을 갱신해줘.
        farthest = torch.max(distance, -1)[1] # distance[b, :]에서 값이 가장 큰 index를 골라서 다음 대표점으로 추가. 현재 대표점들로부터 가장 멀리 떨어진 점 = 가장 덜 커버된 점
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz): # xyz (B,N,3) 전체 점, new_xyz (B,S,3) query/centroid 점
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape # (B, N=2048, 3)
    _, S, _ = new_xyz.shape # (B, S=512, 3)
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # (B,S,N) group_idx를 [0, ..., N-1]로 초기화한 후 (B,S,N) 만들기
    ### group_idx ###
    # tensor([[[   0,    1,    2,  ..., 2045, 2046, 2047],
    #          [   0,    1,    2,  ..., 2045, 2046, 2047],
    #          [   0,    1,    2,  ..., 2045, 2046, 2047],
    #          ...,
    #          [   0,    1,    2,  ..., 2045, 2046, 2047],
    #          [   0,    1,    2,  ..., 2045, 2046, 2047],
    #          [   0,    1,    2,  ..., 2045, 2046, 2047]]], device='cuda:0')
    ### group_idx ###
    sqrdists = square_distance(new_xyz, xyz) # (B,S,N)으로 전체 점과 centroid 점 사이 거리 계산...
    group_idx[sqrdists > radius ** 2] = N # 반경 밖인 점은 index를 N으로 바꿔서 무효 표시
    ### group_idx ###
    # tensor([[[2048, 2048, 2048,  ..., 2048, 2048, 2048],
    #          [2048, 2048, 2048,  ..., 2048, 2048, 2047],
    #          [2048,    1, 2048,  ..., 2048, 2048, 2048],
    #          ...,
    #          [2048, 2048, 2048,  ..., 2048, 2048, 2048],
    #          [2048, 2048, 2048,  ..., 2048, 2048, 2048],
    #          [2048, 2048, 2048,  ..., 2048, 2048, 2048]]], device='cuda:0')
    ### group_idx ###
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # (B,S,K), sort해서 앞쪽에 유효한 index가 오게 한 뒤 앞에서 K개만 자름. N으로 채워진 무효값들이 뒤로 가도록 sort를 쓴다...
    ### group_idx ### 각 group 마다 겹치는 index가 있을 수 있다? 밑에 10을 봐...
    # tensor([[[  9,  25,  43,  ..., 333, 345, 371],
    #          [  8,  45,  51,  ..., 845, 905, 951],
    #          [  1,  38,  44,  ..., 954, 969, 977],
    #          ...,
    #          [ 10,  21,  52,  ..., 340, 342, 355],
    #          [  7,  30,  41,  ..., 538, 563, 575],
    #          [ 10,  32,  52,  ..., 342, 355, 370]]], device='cuda:0')
    ### group_idx ###
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) # (B,S,K), 각 group 당 첫번째 값...
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # 어떤 centroid가 radius 안 점이 거의 없어서 N이 섞이면 group_first로 채움...
    return group_idx # (B,S,K) K=nsample


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False): 
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape # (B, N, 3)
    S = npoint # 512 이번 SA layer에서 대표점 S개를 만들 것.
    fps_idx = farthest_point_sample(xyz, npoint) # (B, S=npoint) 각 batch마다 N개 점 중에서 FPS로 S개를 골라 index list를 만든다. 
    new_xyz = index_points(xyz, fps_idx) # (B, S=npoint, 3) 방금 뽑은 index를 이용해서 실제 대표점 좌표를 가저온다.
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # (B, S=npoint, K=nsample) 대표점 하나 new_xyz[b, s]를 기준으로 원본 점들 xyz[b, 0..N-1] 중에서 거리 <= radius인 점들을 모아서 그 중 최대 K=nsample개를 index로 반환 (부족하면 첫번째 이웃으로 padding). idx[b,s,k]는 b번째 batch, s번째 대표점 주변의 k번째 이웃점의 원본 index 
    grouped_xyz = index_points(xyz, idx) # (B, S=npoint, K=nsample, 3) 이제 index가 아니라 실제 좌표로 바꿈. grouped_xyz[b, s, k]는 s번째 대표점 주변의 k번째 이웃점 좌표.
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # (B, S=npoint, K=nsample, 3) 각 대표점 기준으로 이웃점 좌표를 상대좌표로 변환. 각 대표점이 원점이 되도록 shift. PointNet++는 local geometry를 보고 싶어하기 때문!!!

    if points is not None: # True
        grouped_points = index_points(points, idx) # (B, S=npoint, K=nsample, 35), 같은 이웃 index idx로 feature도 모음
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # (B, S=npoint, K=nsample, 3+35=38)
    else:
        new_points = grouped_xyz_norm
    if returnfps: # False
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points # (B, S=npoint, 3) 실제 대표점 좌표, (B, S=npoint, K=nsample, 3+35=38) 이웃점 상대좌표 + (이웃점 절대좌표 + part emb)


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device) # (B, 1, 3) 대표점 좌표?
    grouped_xyz = xyz.view(B, 1, N, C) # (B, 1, N, 3) 모든 점을 하나의 group으로! 대표점이 1개이고 그 대표점의 이웃은 전체 N개!
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1) # (B, S=1, K=N, 3+256=259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points # (B, 1, 3) 실제 대표점 좌표 all zero, (B, S=1, K=N, 3+256=259), 이웃점 상대좌표 = 절대좌표 + 이웃점 feature


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint # 512 / 128 / None
        self.radius = radius # 0.2 / 0.4 / None
        self.nsample = nsample # 32 / 64 / None
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel # 6 + 32 = 38 / 128 + 3 = 131 / 256 + 3 = 259
        for out_channel in mlp: # [64, 64, 128] / [128, 128, 256] / [256, 512, 1024]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all # False / False / True

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1) # (B, N=2048, 3) / (B, N=512, 3) / (B, N=128, 3)
        if points is not None:
            points = points.permute(0, 2, 1) # (B, N=2048, 3+32=35) / (B, N=512, 128) / (B, N=128, 256)

        if self.group_all: # False / False / True
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # (B, S=512, 3), (B, S=512, K=32, 3+35=38) / (B, S=128, 3), (B, S=128, K=64, 3+128=131) / (B, S=1, 3), (B, S=1, K=128, 3+256=259)  대표점 절대 좌표, 이웃점 상대좌표 + (이웃점 절대좌표 + part emb or 이웃점 feature)
        new_points = new_points.permute(0, 3, 2, 1) # (B, 3+35=38, K=32, S=512) / (B, 3+128=131, K=64, S=128) / (B, 3+256=259, K=128, S=1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points))) # 여기서 conv는 Conv2d(..., kernel_size=1)이니까 K축, S축은 섞지 않고 각 (k,s) 위치의 38dim vector에 동일한 MLP를 적용. 다시 말해 점마다 같은 MLP를 적용해서 점 feature 차원만 점점 커짐.
        # (B, 128, K=32, S=512) / (B, 256, K=64, S=128) / (B, 1024, K=128, S=1) 
        new_points = torch.max(new_points, 2)[0] # (B, 128, S=512) / (B, 256, S=128) / (B, 1024, S=1) dim=2가 K축이니까 각 대표점 s에 대해 K개 점들 중에서 max를 취함. 대표점 s 주변을 보고 대표점마다 하나의 feature vector를 만든 것.
        new_xyz = new_xyz.permute(0, 2, 1) # (B, 3, S=512) / (B, 3, S=128) / (B, 3, S=1) 
        return new_xyz, new_points
         # (B, 3, S=512), (B, 128, S=512) / (B, 3, S=128), (B, 256, S=128) / (B, 3, S=1), (B, 1024, S=1) (permuted) 대표점 절대 좌표, (permuted) 대표점 feature

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel # 1280 / 384 / 128 + 6 + 32 = 166
        for out_channel in mlp: # [256, 256] / [256, 128] / [128, 128, 128]
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1) # (B, 128, 3) / (B, 512, 3) / (B, 2048, 3)
        xyz2 = xyz2.permute(0, 2, 1) # (B, 1, 3) / (B, 128, 3) / (B, 512, 3)

        points2 = points2.permute(0, 2, 1) # (B, 1, 1024) / (B, 128, 256) / (B, 512, 128)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1: # xyz2에 점이 1개면 (sa3) feature도 1개 뿐.
            interpolated_points = points2.repeat(1, N, 1) # (B, 128, 1024) 모든 dense 점에 global feature를 그냥 복붙
        else:
            dists = square_distance(xyz1, xyz2) # (B, 512, 128) / (B, 2048, 512), dense xyz1 각 점과 coarse xyz2 각 점 사이 거리 전부 계산
            dists = dists.clamp_min(0.0)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3] # (B, 512, 3), (B, 512, 3) / (B, 2048, 3), (B, 2048, 3), 가장 가까운 3개 거리와 coarse index

            dist_recip = 1.0 / (dists + 1e-8) # 역거리가중치
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, 512, 1) / (B, 2048, 1)
            weight = dist_recip / norm # (B, 512, 3) / (B, 2048, 3)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) # (B, dense_num=512, coarse_feature_dim=256) / (B, dense_num=2048, coarse_feature_dim=128), dense 각 점마다 가까운 coarse 3개의 feature를 가져와서 weight 곱해서 합침.

        if points1 is not None:
            points1 = points1.permute(0, 2, 1) # (B, 128, 256) / (B, 512, 128) / (B, 2048, 38)
            new_points = torch.cat([points1, interpolated_points], dim=-1) # (B, 128, 256+1024=1280) / (B, 512, 128+256=384) / (B, 2048, 38+128=166)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1) # (B, 256+1024=1280, 128) / (B, 128+256=384, 512) / (B, 38+128=166, 2048)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points))) # Conv1d는 각 점들에 대해 동일한 MLP 적용. point 간 섞지 않음.
        return new_points # (B, 256, 128) / (B, 128, 512) / (B, 128, 2048)