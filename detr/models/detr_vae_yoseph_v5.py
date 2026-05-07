# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer_yoseph_v5 import build_transformer, TransformerEncoder, TransformerEncoderLayer
### yoseph ###
# from .pointnet2_yoseph_v4 import build_pointnet2
### yoseph ###

import numpy as np

import IPython
e = IPython.embed

### yoseph ###
def weighted_pool(feat, weight, eps=1e-6):
    """
    feat:   (B, N, D)
    weight: (B, N)
    return: (B, D)
    """
    w = weight.unsqueeze(-1)
    return (feat * w).sum(dim=1) / (w.sum(dim=1) + eps)


def build_garment_tokens(garment_feat, affordance_score_all, stage_input):
    """
    garment_feat:         (B, 2048, D)
    affordance_score_all: (B, 4, 2048)
    current_stage:        (B,)

    affordance index assumption:
        0: left sleeve
        1: left bottom
        2: right bottom
        3: right sleeve
    """
    B, _, D = garment_feat.shape
    device = garment_feat.device

    left_sleeve = weighted_pool(garment_feat, affordance_score_all[:, 0]) # (B, 128)
    left_bottom = weighted_pool(garment_feat, affordance_score_all[:, 1])
    right_bottom = weighted_pool(garment_feat, affordance_score_all[:, 2])
    right_sleeve = weighted_pool(garment_feat, affordance_score_all[:, 3])

    token1 = torch.zeros(B, D, device=device)
    token2 = torch.zeros(B, D, device=device)

    mask0 = stage_input == 0
    token1[mask0] = right_sleeve[mask0]

    mask1 = stage_input == 1
    token1[mask1] = left_sleeve[mask1]

    mask2 = stage_input == 2
    token1[mask2] = right_bottom[mask2]
    token2[mask2] = left_bottom[mask2]

    return torch.stack([token1, token2], dim=1)
### yoseph ###

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries # 100
        self.camera_names = camera_names # ['top']
        self.transformer = transformer 
        self.encoder = encoder
        hidden_dim = transformer.d_model # 512
        self.action_head = nn.Linear(hidden_dim, state_dim) # (512, 14)
        self.is_pad_head = nn.Linear(hidden_dim, 1) # (512, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim) # Embedding(100, 512)
        if backbones is not None: # True
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1) # 각 (h,w) 위치마다 backbones[0].num_channels -> hidden_dim linear transformation, Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)) 
            self.backbones = nn.ModuleList(backbones) # originally... backbones = [Joiner()]
            self.input_proj_robot_state = nn.Linear(14, hidden_dim) # (14, 512)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # Embedding(1, 512) # # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # (14, 512) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim) # (14, 512) # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # (512, 64) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # (1, 1+1+100, 512) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # (32, 512) # project latent sample to embedding
        ### yoseph ###
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # Embedding(2, 512) # learned position embedding for proprio and latent
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # Embedding(2, 512) # learned position embedding for latent, qpos, env, stage
        pcd_feat_dim = 128

        # self.garment_pcd_encoder = garment_pcd_encoder
        # self.env_pcd_encoder = env_pcd_encoder

        # self.input_proj_garment = nn.Linear(pcd_feat_dim, hidden_dim)
        # self.input_proj_env = nn.Linear(pcd_feat_dim, hidden_dim)

        # self.stage_embed = nn.Embedding(3, hidden_dim)
        # self.null_garment_token = nn.Parameter(torch.zeros(hidden_dim)) # (512)

        # self.stage_head = nn.Linear(hidden_dim, 3)
        ### yoseph ###
    def forward(self, 
                qpos, 
                image, 
                ### yoseph ###
                # env_pcd=None,
                # garment_pcd=None,
                # stage_input=None,
                # affordance_score=None,
                ### yoseph ###
                actions=None, 
                is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training: # True
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (B, L, 512) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos) # (B, 512) # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1) # (B, 1, 512)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, 512) # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (B, 1, 512) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (B, L+2, 512) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (L+2, B, 512) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # (B, 2) all False # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1) # (B, L+2) # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach() # (1, L+2, 512)
            pos_embed = pos_embed.permute(1, 0, 2) # (L+2, 1, 512) # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad) # (L+2, B, 512)
            encoder_output = encoder_output[0] # (B, 512) # take cls output only!!!
            latent_info = self.latent_proj(encoder_output) # (B, 64)
            mu = latent_info[:, :self.latent_dim] # (B, 32)
            logvar = latent_info[:, self.latent_dim:] # (B, 32)
            latent_sample = reparametrize(mu, logvar) # (B, 32)
            latent_input = self.latent_out_proj(latent_sample) # (B, 512)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device) # (B, 32)
            latent_input = self.latent_out_proj(latent_sample) # (B, 512)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names): # ['top']
                ### yoseph ###
                # features, pos = self.backbones[0](image[:, cam_id]) # class Joiner의 def forward로 들어감 # HARDCODED 
                features, pos = self.backbones[cam_id](image[:, cam_id])
                ### yoseph ###
                features = features[0] # (B, 512, 15, 20) [0] 하는 거 그냥 걱정하지 말고 받아들여. # take the last layer feature 
                pos = pos[0] # (1, 512, 15, 20)
                all_cam_features.append(self.input_proj(features)) # (B, 512, 15, 20)
                all_cam_pos.append(pos) 
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos) # (B, 512)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3) # (B, 512, 15, 20*3)
            pos = torch.cat(all_cam_pos, axis=3) # (1, 512, 15, 20*3)
                        
            ### yoseph ###
            # garment_feat = self.garment_pcd_encoder(garment_pcd) # (B, 2048, 128)... little bit slow...
            # garment_tokens = build_garment_tokens( # (B, 2, 128)
            #     garment_feat=garment_feat,
            #     affordance_score_all=affordance_score,
            #     stage_input=stage_input
            # )
    
            # garment_tokens = self.input_proj_garment(garment_tokens) # (B, 2, 512)

            # null = self.null_garment_token.view(1, 1, -1).expand(bs, 1, -1) # (B, 1, 512)

            # mask_null_slot2 = stage_input != 2 # (B)
            # garment_tokens[:, 1:2] = torch.where(
            #     mask_null_slot2.view(bs, 1, 1),
            #     null,
            #     garment_tokens[:, 1:2],
            # )

            # env_feat = self.env_pcd_encoder(env_pcd) # (B, 128)
            # env_input = self.input_proj_env(env_feat) # (B, 512)

            # stage_input_token = self.stage_embed(stage_input) # (B, 512)

            extra_inputs = torch.stack( # (6, B, 512)
                [
                    latent_input,
                    proprio_input,
                    # env_input,
                    # stage_input_token,
                    # garment_tokens[:, 0],
                    # garment_tokens[:, 1],
                ],
                dim=0,
            )
            # 엥? 이거 맞아? decoder에 layer 7개 있는데 결국 첫번째 layer output만 쓴다구? [-1]로 고쳐야 하나?
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0] # (B, 100, 512), self.query_embed Embedding(100, 512), self.additional_pos_embed Embedding(2, 512)
            hs = self.transformer( # (B, 50, 512)
                src=src,
                mask=None,
                query_embed=self.query_embed.weight,
                pos_embed=pos,
                extra_inputs=extra_inputs,
                extra_pos_embed=self.additional_pos_embed.weight,
            )[-1]
            ### yoseph ###
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs) # (B, 50, 14)
        is_pad_hat = self.is_pad_head(hs) # (B, 100, 1)
        ### yoseph ###
        # stage_feature = hs.mean(dim=1) # (B, 512)
        # stage_logits = self.stage_head(stage_feature) # (B, 3)
        return a_hat, is_pad_hat, [mu, logvar]
        # return a_hat, is_pad_hat, [mu, logvar] # mu & logvar are from latent_info
        ### yoseph ###


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 512
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 3200
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None # None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode
    ### yoseph ###
    # env_pcd_encoder = build_pointnet2()
    ### yoseph ###
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    ### yoseph ###
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    # backbone = build_backbone(args) # detr/models/backbone.py backbone + position_embedding
    # backbones.append(backbone)
    ### yoseph ###

    transformer = build_transformer(args) # detr/models/transformer.py 

    encoder = build_encoder(args) # HERE!!! 내 생각엔 transformer.encoder랑 같은 거 같아...

    model = DETRVAE(
        ### yoseph ###
        # garment_pcd_encoder,
        # env_pcd_encoder,
        ### yoseph ###
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

### encoder ###
# TransformerEncoder(
#   (layers): ModuleList(
#     (0-3): 4 x TransformerEncoderLayer(
#       (self_attn): MultiheadAttention(
#         (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
#       )
#       (linear1): Linear(in_features=512, out_features=3200, bias=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#       (linear2): Linear(in_features=3200, out_features=512, bias=True)
#       (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#       (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#       (dropout1): Dropout(p=0.1, inplace=False)
#       (dropout2): Dropout(p=0.1, inplace=False)
#     )
#   )
# )
### encoder ###