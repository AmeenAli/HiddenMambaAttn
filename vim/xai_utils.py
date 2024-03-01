import torch
import numpy as np


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def generate_raw_attn(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    all_layer_attentions = []
    cls_pos = 98
    for layeridx in range(len(model.layers)):
        attn_heads = model.layers[layeridx].mixer.xai_b
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    p = torch.cat(all_layer_attentions[start_layer:], dim=0).mean(dim=0).unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits


def generate_mamba_attr(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    index = np.argmax(logits.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    all_layer_attentions = []
    cls_pos = 98
    for layeridx in range(len(model.layers)):
        attn_heads = model.layers[layeridx].mixer.xai_b.clamp(min=0)
        s = model.layers[layeridx].get_gradients().squeeze().detach() #[1:, :].clamp(min=0).max(dim=1)[0].unsqueeze(0)
        s = s.clamp(min=0).max(dim=1)[0].unsqueeze(0)
        s = (s - s.min()) / (s.max() - s.min())
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        fused = avg_heads * s
        all_layer_attentions.append(fused)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer)
    p = rollout[0 , cls_pos , :].unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits
 

def generate_rollout(model, image, start_layer=15, num_layers=24):
    image.requires_grad_()
    logits = model(image)    
    all_layer_attentions = []
    cls_pos = 98
    for layer in range(num_layers):
        attn_heads = model.layers[layer].mixer.xai_b
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    p = rollout[0 , cls_pos , :].unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits
