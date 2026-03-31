import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override) # read detr/main.py 너무 깊숙히 들어가네...
        self.model = model # CVAE decoder가 아니라 전체 아니야? # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight'] # 10
        print(f'KL Weight {self.kl_weight}')

    ### yoseph ###
    def __call__(self, qpos, image, env_feature=None, garment_feature=None, actions=None, is_pad=None):
    # def __call__(self, qpos, image, actions=None, is_pad=None):
    ### yoseph ###
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet RGB statistics
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image) # (B, 1, 3, 480, 640)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries] # (B, 100, 14)
            is_pad = is_pad[:, :self.model.num_queries] # (B, 100)
            # a_hat (B, 100, 14), mu (B, 32), logvar (B, 32)
            ### yoseph ###
            # a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, env_feature, garment_feature, actions, is_pad)
            ### yoseph ###
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar) # (1)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none') # (B, 100, 14)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1 # just scalar
            loss_dict['kl'] = total_kld[0] # just scalar
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight # * 10
            return loss_dict
        else: # inference time
            ### yoseph ###
            # a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            a_hat, _, (_, _) = self.model(qpos, image, env_state, env_feature, garment_feature)
            ### yoseph ###
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4: # False
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4: # False
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # (B, 32)
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
