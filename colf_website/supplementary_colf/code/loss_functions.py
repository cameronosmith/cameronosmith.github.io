# import pyshearlab
from skimage.filters import gabor_kernel
import util
import matplotlib.pyplot as plt #debug
#import sklearn
def save(x,suffix=""):
    plt.imshow(x)

#from kornia.filters import spatial_gradient
#import kornia
import random
import numpy as np
import torch
from torch.utils.data        import DataLoader
import torch.nn.functional as F
import diff_operators
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import geometry
from einops import repeat,rearrange
from torch import autograd

from pdb import set_trace as pdb #debug

import conv2d_gradfix

import torchgeometry

# ideas:self sup feature attention vs soft seg consistency loss
#       apply attention weights on rest of query images for seg/key consistency
#       contrastive loss on latent codes from slot encoder

import lpips
import piq

if 'loss_fn_alex' not in globals():
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
def tmp(model_out,gt):
    print( rgb(model_out,gt) )
    print( rebut_seg(model_out,gt) )
    print( rebut_seg_attn(model_out,gt) )
    #print( ssim(model_out,gt) )
    #print( latent_penalty(model_out,gt) )
    #print( latent_penalty(model_out,gt) )
    #print( lpips_vgg(model_out,gt) )
    return
import kornia

def rebut_seg(model_out,gt):
    gtseg=torch.stack(gt["seg"])
    fg=(gtseg!=0).float()
    gtseg=torch.stack([gtseg==i for i in gtseg.unique()[1:]]).float()
    seg=model_out["seg"]
    return ((seg-gtseg).square()*fg).mean()
def rebut_seg_attn(model_out,gt):
    gtseg=torch.stack(gt["seg"])[:,0].squeeze(-1)
    #gtseg=F.interpolate(gtseg.unflatten(-1,(64,64)).unsqueeze(1),(128,128),
    #        mode="nearest").flatten(-2,-1).squeeze()
    fg=(gtseg!=0).float()
    gtseg=torch.stack([gtseg==i for i in gtseg.unique()[1:]]).float()
    seg=model_out["attn"].permute(1,0,2)#[:,:,None]#*bg
    return ((gtseg-seg).square()*fg).mean()
    #return (gtseg[:,None].float().squeeze(-1)-seg[None].float()).abs().min(1)[0].mean()
def invmask_comp(model_out,gt):
    seg=model_out["coarse_seg"][:,:,0].squeeze(-1).permute(1,0,2)
    seg=F.interpolate(seg.unflatten(-1,(64,64)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    invmasks=gt["inv_mask"][0]
    blur=lambda x,y=3:kornia.filters.blur_pool2d(x[None,None],y,1)[0][0]
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda()
    loss=0
    #for invmask,alpha,fgattn in zip(invmasks,model_out["alpha"][:,0],attn[:,1:]):
    for invmask,alpha,fgattn in zip(invmasks,gt["fg"][:,0],seg[:,1:]):
        blurfg1=blur(alpha.view(32,32)*mask,11)
        maxweight,maxidx=blurfg1.flatten().max(0)
        maxattn=fgattn[fgattn[:,maxidx].max(0)[1]]
        loss+=(mask.flatten()*(invmask.flatten()>0).float()*maxattn.flatten()).mean()
    return loss
def invmask_attn(model_out,gt):
    invmasks=gt["inv_mask"][0]
    blur=lambda x,y=3:kornia.filters.blur_pool2d(x[None,None],y,1)[0][0]
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda()
    attn=F.interpolate(model_out["attn"].unflatten(-1,(128,128)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    loss=0
    #for invmask,alpha,fgattn in zip(invmasks,model_out["alpha"][:,0],attn[:,1:]):
    for invmask,alpha,fgattn in zip(invmasks,gt["fg"][:,0],attn[:,1:]):
        blurfg1=blur(alpha.view(32,32)*mask,11)
        maxweight,maxidx=blurfg1.flatten().max(0)
        maxattn=fgattn[fgattn[:,maxidx].max(0)[1]]
        loss+=(mask.flatten()*(invmask.flatten()>0).float()*maxattn.flatten()).mean()
    return loss
def transient_comp_fg(model_out,gt):
    fg_pres=gt["fg"][:,0].detach()
    seg=model_out["coarse_seg"][:,:,0].squeeze(-1).permute(1,0,2)
    seg=F.interpolate(seg.unflatten(-1,(64,64)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda().flatten()[None,...,None].expand(seg.size(0),-1,-1)
    fg_attn=seg[:,1:].sum(1)
    return (mask*(1-fg_attn)*fg_pres).mean()
def transient_comp_bg(model_out,gt):
    bg_pres=1-gt["fg"][:,:1].detach()
    seg=model_out["coarse_seg"][:,:,0].squeeze(-1).permute(1,0,2)
    seg=F.interpolate(seg.unflatten(-1,(64,64)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    scaled_bg_pres=1/(1+torch.exp(15-20*bg_pres))
    bg_attn=seg[:,:1]
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda().flatten()[None,...,None].expand(seg.size(0),-1,-1)[:,None]
    return (mask*(1-seg)*scaled_bg_pres).mean()
def ssim(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    fake_samples = model_out["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    real_samples = gt["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    return 1-piq.ssim(fake_samples*.5+.5,real_samples*.5+.5)
def latent_penalty(model_out,gt):
    return model_out["fg_latent"].square().mean()
def transient_attn_fg(model_out,gt):
    #fg_pres=model_out["alpha"][:,0].detach()
    fg_pres=gt["fg"][:,0].detach()
    attn=F.interpolate(model_out["attn"].unflatten(-1,(128,128)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    fg_attn=attn[:,1:].sum(1)
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda().flatten()[None,...,None].expand(fg_attn.size(0),-1,-1)
    return (mask*(1-fg_attn)*fg_pres).mean()
def transient_attn_bg(model_out,gt):
    #bg_pres=1-model_out["alpha"][:,:1].detach()
    bg_pres=1-gt["fg"][:,:1].detach()
    attn=F.interpolate(model_out["attn"].unflatten(-1,(128,128)),(32,32)
                       ).flatten(-2,-1).unsqueeze(-1)
    scaled_bg_pres=1/(1+torch.exp(15-20*bg_pres))
    bg_attn=attn[:,:1]
    i,j=10,32;mask=np.pad(np.ones((j-i,j-i)), pad_width=i//2, mode='constant', constant_values=0)
    mask=torch.from_numpy(mask).cuda().flatten()[None,...,None].expand(bg_attn.size(0),-1,-1)[:,None]
    return (mask*(1-bg_attn)*scaled_bg_pres).mean()
def transient(model_out,gt):
    return model_out["alpha"].mean()
def fg_emerg(model_out,gt):
    mask=torch.ones(64,64).cuda()
    mask[:10]=0
    mask[-10:]=0
    mask[:,:10]=0
    mask[:,-10:]=0
    bg=model_out["coarse_rgbs"][0][:,0].permute(0,2,1).unflatten(-1,(64,64))
    mask=mask[None].expand(bg.size(0),-1,-1)
    gt=gt["rgb"][:,0].permute(0,2,1).unflatten(-1,(64,64))
    blur=lambda x:kornia.filters.blur_pool2d(x,9,1)
    targ=mask*(blur(gt)-blur(bg)).sum(1).square()
    fg_seg=model_out["coarse_seg"][1:].sum(0)[:,0].squeeze(-1).unflatten(-1,(64,64))
    fg_attn=model_out["slot_attn"][:,1:].sum(1).unflatten(-1,(128,128))
    fg_attn=F.interpolate(fg_attn[:,None],(64,64)).squeeze(1)
    segloss=((1-fg_seg)*targ).mean()
    attnloss=((1-fg_attn)*targ).mean()
    
    # Where bg recon accuracy > .8, supervise bg attends there
    bg_attn=model_out["slot_attn"][:,:1].sum(1).unflatten(-1,(128,128))
    bg_attn=F.interpolate(bg_attn[:,None],(64,64)).squeeze(1)
    bg_attn_targ=(1-((gt*.5+.5)-(bg*.5+.5)).abs()).mean(1)>.95
    bg_attnloss=((1-bg_attn)*bg_attn_targ.float()).mean()/100

    bg_attnloss2=((blur(bg)-blur(gt)).abs().mean(1)*bg_attn).mean()
    #print(bg_attnloss,segloss,attnloss,bg_attnloss2)

    return segloss+attnloss+bg_attnloss+bg_attnloss2
def onehot_comp(model_out,gt):
    return ( 1-model_out["fine_seg"].max(0)[0] ).mean()
def black_bg(model_out,gt):
    bg_weight=1-model_out["per_phi_seg"]
    return (model_out["coarse_rgbs"][1:]*bg_weight).square().mean()
def ray_attn_seg(model_out,gt):
    bg_weight=1-model_out["per_phi_seg"].flatten(0,2)
    return (model_out["ray_attn"]*bg_weight).square().mean()
def rot_consistency(model_out,gt):
    pts=model_out["zrots"].flatten(0,1)
    return (pts[:,None]-pts[:,:,None]).square().mean()
def center_pts(model_out,gt):
    pts=model_out["center_pts"].flatten(0,1)
    return (pts[:,None]-pts[:,:,None]).square().mean()
def canonical(model_out,gt):
    x=model_out["sym_rgb"].detach().unflatten(1,(64,64))
    flipped=torch.flip(x,[2])
    return (flipped-x).square().mean()
    
def fgloss(model_out,gt):
    return model_out["fgloss"]
def gt_mask_attn(model_out,gt):
    attn=model_out["A_attn"].permute(1,0,2)
    mask_flat=torch.stack(gt["gt_mask"],1)[:,0].squeeze(-1)
    mask = torch.stack([mask_flat==i for i in mask_flat.unique()]).float()
    return (mask-attn).square().mean()
    """
    attn=model_out["A_attn"]
    bgattn=attn[:,0]
    fgattn=attn[:,1:].permute(1,0,2)
    mask=torch.stack(gt["gt_maskattn"],1)[:,0].squeeze(-1)
    bgmask=(mask==0).float()
    fgmask=(mask==mask.unique()[-1]).float()
    fg_loss=(fgmask[None]-fgattn).square().min(0)[0].mean()
    bg_loss=(bgmask-bgattn).square().mean()
    return fg_loss+bg_loss
    """
def gt_mask_comp(model_out,gt):
    seg=model_out["coarse_seg"].flatten(1,2).squeeze(-1)
    mask_flat=torch.stack(gt["gt_mask"],1).flatten(0,1).squeeze(-1)
    mask = torch.stack([mask_flat==i for i in mask_flat.unique()]).float()
    return (mask-seg).square().mean()
    """
    seg=model_out["coarse_seg"].flatten(1,2).squeeze(-1)
    bgseg=seg[0]
    fgseg=seg[1:]
    mask=torch.stack(gt["gt_mask"],1).flatten(0,1).squeeze(-1)
    bgmask=(mask==0).float()
    fgmask=(mask==mask.unique()[-1]).float()
    fg_loss=(fgmask[None]-fgseg).square().min(0)[0].mean()
    bg_loss=(bgmask-bgseg).square().mean()
    return fg_loss+bg_loss
    """
def attn_teacher(model_out,gt):
    coarse_attn=gt["coarse_out"]["attn"][:,1:].detach()
    fine_attn=model_out["attn"][:,1:]
    return (coarse_attn*(coarse_attn-fine_attn).square()).mean()
def mask_teacher(model_out,gt):
    coarse_seg=gt["coarse_out"]["seg"].detach()
    #coarse_weights = (1/(1+torch.exp(35-40*coarse_seg))+
    #                  1/(1+torch.exp(-5+40*coarse_seg)))
    return (coarse_seg-model_out["seg"]).square().mean()

def l1(model_out,gt):
    return F.l1_loss(model_out["rgb"],gt["rgb"])
def rgb(model_out,gt):
    return F.mse_loss(model_out["rgb"],gt["rgb"])
def rgb_coarse(model_out,gt):
    return F.mse_loss(model_out["coarse_rgb"],gt["rgb"])
def rgb_fine(model_out,gt):
    return F.mse_loss(model_out["fine_rgb"],gt["rgb"])

def tmp_gan(model_out,gt,model,iter_):
    return

def lpips_vgg(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    pred_rgb,gt_rgb=[src["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
                for src in (model_out,gt)]
    return loss_fn_vgg(pred_rgb,gt_rgb).mean()

def lpips_loss(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    pred_rgb,gt_rgb=[src["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
                for src in (model_out,gt)]
    return loss_fn_alex(pred_rgb,gt_rgb).mean()

def rotation_angle(r1,r2):
    R = torch.bmm(r1, r2.permute(0, 2, 1))
    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    return phi_cos

def pose_est(model_out,gt):
    gt_rel=gt["cam2world"][:,1].inverse()@gt["cam2world"][:,0]
    est1,est2 = model_out["poses"].unflatten(1,(gt_rel.size(0),2))
    est1_rel=est1[:,1].inverse()@est1[:,0]
    est2_rel=est2[:,1].inverse()@est2[:,0]
    rot1=rotation_angle(est1_rel[:,:3,:3],gt_rel[:,:3,:3])
    rot2=rotation_angle(est2_rel[:,:3,:3],gt_rel[:,:3,:3])
    trans1=F.mse_loss(gt_rel[:,:3,-1],est1_rel[:,:3,-1])
    trans2=F.mse_loss(gt_rel[:,:3,-1],est2_rel[:,:3,-1])
    return trans1.mean()+trans2.mean()+rot1.mean()+rot2.mean()

def attn_conn_comp_loss2(model_out,gt):
    slots = model_out["A_attn"]
    labels = gt["comp"][:,0]
    objs = labels==labels.unique()[None,None]
    inv_objs = (~objs).float()
    return (slots.unsqueeze(-1)*inv_objs[:,None]).sum(-2).min(1)[0].mean()/slots.size(-1)

def attn_conn_comp_loss_unique(model_out,gt):
    # GOAL:
    # - Only apply to FG elements
    # - Only use largest FG label obj (second largest) 
    #   to prevent one slot from getting conflicting losses
    labels = gt["comp"].flatten(0,1)
    obj = labels==1
    inv = labels!=1
    fg = model_out["all_A_attn"][:,1:]
    idx= (obj[:,None].squeeze(-1)*fg).sum(-1).max(1)[1]
    fidx=torch.arange(fg.size(0))*fg.size(1)+idx.cpu()
    targ_slot = fg.flatten(0,1)[fidx]
    return (targ_slot*inv.squeeze(-1)).square().mean()

def comp_conn_comp_loss_unique(model_out,gt):
    # GOAL:
    # - Only apply to FG elements
    # - Only use largest FG label obj (second largest) 
    #   to prevent one slot from getting conflicting losses
    labels = gt["comp"].flatten(0,1)
    obj = labels==1
    inv = labels!=1
    attn = rearrange(model_out["soft_seg"],"p b q pix 1 -> (b q) p pix")
    fg = attn[:,1:]
    idx= (obj[:,None].squeeze(-1)*fg).sum(-1).max(1)[1]
    fidx=torch.arange(fg.size(0))*fg.size(1)+idx.cpu()
    targ_slot = fg.flatten(0,1)[fidx]
    return (targ_slot*inv.squeeze(-1)).square().mean()

def attn_conn_comp_loss(model_out,gt):
    # note we already perform matching here so need to change anything for all_attn
    labels = gt["comp"].flatten(0,1)
    slots = model_out["all_A_attn"]
    objs = labels==labels.unique()[None,None]
    obj_idxs = (slots.unsqueeze(-1)*objs[:,None]).sum(2).max(-1)[1]
    unclaimed_objs = labels!=obj_idxs[:,None]
    return (unclaimed_objs.permute(0,2,1)*slots).mean()
    """
    loss = 0
    labels = gt["comp"][:,0]
    slots = model_out["A_attn"]
    objs = labels==labels.unique()[None,None]
    obj_idxs = (slots.unsqueeze(-1)*objs[:,None]).sum(2).max(-1)[1]
    unclaimed_objs = labels!=obj_idxs[:,None]
    loss += (unclaimed_objs.permute(0,2,1)*slots).mean()
    return loss
    """

def attn_comp_consistency(model_out,gt):
    #  first slot is unique matching
    loss = 0
    attn = model_out["A_attn"].detach()
    comp = model_out["soft_seg"][:,:,0].squeeze(-1).permute(1,0,2)
    loss +=  (attn-comp).square().mean()
    """
    # for rest of slots have to try all matches
    attn = model_out["all_A_attn"].detach()
    comp = model_out["soft_seg"].flatten(1,2).squeeze(-1).permute(1,0,2)
    loss += (attn[:,:,None]-comp[:,None]).square().min(dim=2)[0].mean()
    """

    return loss

def attn_bg_presence(model_out,gt):

    noflow = (gt["comp"]==0).float().flatten(0,1)
    bg_attn = model_out["all_A_attn"][:,0,...,None] #only bg (first slot) 
    return F.mse_loss(bg_attn,noflow)

    """
    bg_attn = model_out["all_A_attn"][:,0] #only bg (first slot) 
    flow_mag = ( 1e2*gt["flow"].flatten(0,1).norm(dim=-1) ).clip(0,256)/256
    flow_mag[flow_mag<.2]=0
    occupy_weights = 1-flow_mag
    tmp = ( torch.ones_like(bg_attn)-bg_attn ).abs()*occupy_weights
    return tmp.mean()
    """
    """
    bg_attn = model_out["A_attn"][:,0] #only bg (first slot) 
    flow_mag = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256)/256
    flow_mag[flow_mag<.2]=0
    occupy_weights = 1-flow_mag
    tmp = ( torch.ones_like(bg_attn)-bg_attn ).abs()*occupy_weights
    return tmp.mean()
    """

def comp_bg_presence(model_out,gt):
    # Trying at all MV not just query
    all_comp=True
    if all_comp:
        bg_comp = model_out["soft_seg"][0].squeeze(-1) #only bg (first slot) 
        flow_mag = ( 1e2*gt["flow"].norm(dim=-1) ).clip(0,256)/256
        flow_mag[flow_mag<.2]=0
        occupy_weights = 1-flow_mag
        tmp = ( torch.ones_like(bg_comp)-bg_comp ).abs()*occupy_weights
        return tmp.mean()
    else:
        bg_comp = model_out["soft_seg"][0,:,0].squeeze() #only bg (first slot) 
        flow_mag = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256)/256
        flow_mag[flow_mag<.2]=0
        occupy_weights = 1-flow_mag
        tmp = ( torch.ones_like(bg_comp)-bg_comp ).abs()*occupy_weights
        return tmp.mean()

def comp_bg_noflow(model_out,gt):
    all_comp=True
    if all_comp:
        bg_comp = model_out["soft_seg"][0].squeeze(-1) #only bg (first slot) 
        flow_mag = ( 1e2*gt["flow"][:,:].norm(dim=-1) ).clip(0,256)/256
        flow_mag[flow_mag<.2]=0
        bg_flow = bg_comp*flow_mag
        return bg_flow.mean()
    else:
        # Trying at all MV not just query
        bg_comp = model_out["soft_seg"][0,:,0].squeeze() #only bg (first slot) 
        flow_mag = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256)/256
        flow_mag[flow_mag<.2]=0
        bg_flow = bg_comp*flow_mag
        #torch.Size([2, 4096]), torch.Size([2, 4096])
        return bg_flow.mean()


def attn_bg_noflow(model_out,gt):

    non_bg = (gt["comp"]!=0).float().flatten(0,1)
    bg_attn = model_out["all_A_attn"][:,0,...,None] #only bg (first slot) 
    return (non_bg*bg_attn).square().mean()

    """
    flow_mag = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256)/256
    flow_mag[flow_mag<.2]=0 #may need to change to .35 for this dataset
    bg_flow = bg_attn*flow_mag
    return bg_flow.mean()
    """

    """
    bg_attn = model_out["all_A_attn"][:,0] #only bg (first slot) 
    flow_mag = ( 1e2*gt["flow"].flatten(0,1).norm(dim=-1) ).clip(0,256)/256
    flow_mag[flow_mag<.2]=0
    bg_flow = bg_attn*flow_mag
    return bg_flow.mean()
    """

def flow_cos_sim(model_out,gt):

    norm = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256).unsqueeze(1)/256
    flow = gt["flow"][:,0].permute(0,2,1)

    attn = model_out["A_attn"]
    tempd_attn = (5*attn).softmax(dim=1)

    mean_flow = ( (10*tempd_attn).softmax(-1).unsqueeze(2)*flow[:,None] ).sum(-1)

    tmp1=repeat(flow,"b uv xy -> b s uv xy",s=attn.size(1))
    tmp2=repeat(mean_flow,"b s uv -> b s uv xy",xy=flow.size(-1))
    cos = F.cosine_similarity(tmp1,tmp2,dim=2)
    sim=((cos+.3)*attn*norm).clip(0,1)

    weighted_loss = (torch.ones_like(sim)-sim).abs()*tempd_attn
    return weighted_loss.mean()
    
def attn_flowmag_var(model_out,gt):

    norm = ( 1e2*gt["flow"][:,0].norm(dim=-1) ).clip(0,256).unsqueeze(1)/256

    attn = model_out["A_attn"]
    tempd_attn = (5*attn).softmax(dim=1)

    slot_mean_flowmag = ( (1e1*tempd_attn).softmax(2)*norm ).sum(-1)
    
    slot_flowmag_var = tempd_attn*(norm-slot_mean_flowmag.unsqueeze(-1)).abs()

    return slot_flowmag_var.mean()

def bg_consistency_loss(model_out,gt):
    bg_context = model_out["slot_z"][:,0]
    bg_query   = model_out["query_slot_z"][:,0]
    return (F.normalize(bg_context)-F.normalize(bg_query)).square().mean()
    #sim = (1+F.cosine_similarity(bg_context,bg_query))/2
    #return F.mse_loss(sim,torch.ones_like(sim))

def fg_wakeup_loss(model_out,gt):
    grid=np.meshgrid(np.linspace(-1.0, 1.0, 64),np.linspace(-1.0, 1.0, 64))
    grid=torch.stack([torch.from_numpy(x) for x in grid]).cuda()
    circle = grid[0]**2+grid[1]**2
    square = (grid> -.7)&(grid<.7)
    square = square[0]*square[1]
    should_occupy     = (1/(square.float()+(~square*circle).float())).flatten()
    should_not_occupy = (circle.exp()-1).flatten()
    return ( (should_occupy-model_out["soft_seg"].squeeze()).abs().mean() +
             (should_not_occupy*model_out["soft_seg"].squeeze()).mean() )

def gen_gan(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    fake_samples = model_out["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    f_preds = model_out["disc"](fake_samples).squeeze()
    return torch.mean(nn.Softplus()(-f_preds))
def gen_gan(model_out,gt):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    fake_samples = model_out["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    f_preds = model_out["disc"](fake_samples).squeeze()
    return torch.mean(nn.Softplus()(-f_preds))

def R1Penalty(real_img,disc):

    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_img = torch.autograd.Variable(real_img, requires_grad=True)
    real_logit = disc(real_img)
    real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                     grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                     create_graph=True, retain_graph=True)[0]
    r1_penalty = real_grads.square().sum()
    return r1_penalty

def disc_gan_r1(model_out,gt):
    return disc_gan(model_out,gt,True)
def disc_gan(model_out,gt,r1=False):
    imsl=int(gt["rgb"].size(-2)**(1/2))
    fake_samples = model_out["rgb"].detach().flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    real_samples = gt["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsl,imsl))
    r_preds = model_out["disc"](real_samples).squeeze()
    f_preds = model_out["disc"](fake_samples).squeeze()

    loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

    if r1:
        r1_penalty = R1Penalty(real_samples.detach(),model_out["disc"]) * 5
        return r1_penalty

    return loss

    """
    criterion = F.binary_cross_entropy_with_logits
    # calculate the real loss:
    real_loss = criterion(
        r_preds.squeeze(),
        torch.ones_like(r_preds))

    # calculate the fake loss:
    fake_loss = criterion(
        f_preds,
        torch.zeros_like(f_preds))
    return (real_loss + fake_loss) / 2
    """

def gan_r1_loss(model_out,gt,model,iter_):
    disc=model.discriminator
    #disc.zero_grad()
    with conv2d_gradfix.no_weight_gradients():
        real_img=gt["input_rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(64,64))
        real_img.requires_grad=True
        real_pred = disc(real_img)
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def attn_seg_consistency(model_out,gt):
    return (model_out["soft_seg"].squeeze()[:,:,0]-
            model_out["slot_attn"].squeeze()).square().mean()
def onehot_attn(model_out,gt):
    attn = model_out["slot_attn"]
    return F.mse_loss(attn.max(-1)[0],torch.ones_like(attn[...,0]))
def attn_spatial_variance(model_out,gt):

    attn = model_out["A_attn"].permute(1,0,2)[1:]
    spatial_attn = (1e2*attn).softmax(-1)
    uv = gt["uv"][None,:,0]
    mean_uv = (spatial_attn[...,None]*uv).sum(-2,keepdim=True)
    uv_var = ( spatial_attn[...,None]*(mean_uv-uv).abs().mean(-1,True) ).squeeze(2)
    return uv_var.mean()

    """
    from summaries import write_img

    attn = model_out["slot_attn"][1:]
    attn_raw = attn.permute(0,1,3,2).repeat(1,1,1,3)

    pixnorm_attn = (1e2*attn).softmax(-1)
    write_img(pixnorm_attn.permute(1,0,2,3).flatten(0,1),None,0,"pixnorm_attn",write_local=True,nrow=4)
    
    # Write tempd attn
    n_phi = attn.size(0)    
    uv = gt["uv"][None]
    mean_uv = (pixnorm_attn[...,None]*uv).sum(-2,keepdim=True)
    uv_var = ( (mean_uv-uv).abs().mean(-1,True) ).squeeze(2)
    write_img(uv_var.squeeze().permute(1,0,2).flatten(0,1).unsqueeze(1),None,0,"uv_var",write_local=True,nrow=4)
    uv_var = ( attn[...,None]*(mean_uv-uv).abs().mean(-1,True) ).squeeze(2)
    write_img(uv_var.squeeze().permute(1,0,2).flatten(0,1).unsqueeze(1),None,0,"attn_uv_var",write_local=True,nrow=4)

    # Write raw attn
    n_phi = attn.size(0)
    attn_ = attn.permute(0,1,3,2).repeat(1,1,1,3)
    rgb   = gt["rgb"][:,0].unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
    tmp = attn_.unsqueeze(1)
    tmp = tmp.permute(2,0,1,3,4).flatten(0,2).permute(0,2,1)
    tmp = F.interpolate(tmp.unflatten(-1,(64,64,)),scale_factor=4).flatten(-2,-1)
    write_img(tmp, None, 0, "raw_attn",
            nrow=n_phi, write_local=True,normalize=True)

    # Write attn and rgb combo
    n_phi = attn.size(0)
    rgb   = gt["rgb"][:,0].unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
    applied_attn = attn_raw*rgb
    tmp = torch.stack((attn_raw,applied_attn),1)
    tmp = tmp.permute(2,0,1,3,4).flatten(0,2).permute(0,2,1)
    tmp = F.interpolate(tmp.unflatten(-1,(64,64,)),scale_factor=4).flatten(-2,-1)
    write_img(tmp, None, 0, "attn_rgb",
            nrow=2*n_phi, write_local=True,normalize=False)
    z
    """

    attn=(1e5*model_out["slot_attn"])[1:].softmax(-1).unsqueeze(-1)
    uv = gt["uv"][None]
    mean_uv = (attn*uv).sum(-2,keepdim=True)
    raw_attn=(1e5*model_out["slot_attn"])[1:].softmax(-1).unsqueeze(-1)
    uv_var = raw_attn*(mean_uv-uv).square().sum(-1,True)
    pdb()
    return uv_var.mean()

def seg_spatial_variance(model_out,gt):
    #softmax over pix
    from summaries import write_img
    wr = lambda x,y,z=8 :write_img(x,None,0,y,z)
    seg=(1e7*model_out["soft_seg"]).squeeze(2)[1:].softmax(-2)
    uv = gt["uv"][None]
    mean_uv = (seg*uv).sum(-2,keepdim=True)

    raw_seg=(1e0*model_out["soft_seg"]).squeeze(2)[1:].softmax(-2)
    uv_var = raw_seg*(mean_uv-uv).square().sum(-1,True)
    return uv_var.mean()

    # VIS
    diff_plain = (mean_uv-uv).square().sum(-1,True)

    n_phi = model_out["soft_seg"].size(0)
    attn  = model_out["slot_attn"].permute(0,1,3,2).repeat(1,1,1,3)
    soft_seg = model_out["soft_seg"].squeeze(2).repeat(1,1,1,1,3)

    write_img(diff_plain[0].flatten(0,1).permute(0,2,1)[:,:1], None, 0, "diffplain",nrow=2)
    write_img(uv_var[0].flatten(0,1).permute(0,2,1)[:,:1], None, 0, "uvvar",nrow=2)
    write_img(seg[0].flatten(0,1).permute(0,2,1), None, 0, "seg",nrow=2)
    for i in range(1,2):
        seg = soft_seg[i].flatten(0,1)
        #img=torch.stack((seg,model_out["rgbs"][i].flatten(0,1)*seg),1).flatten(0,1).permute(0,2,1)
        img=seg.unsqueeze(1).flatten(0,1).permute(0,2,1)
        write_img(img, None, 0, f"tmp_Slot_{i}_seg",nrow=2)

    zz
    uv_var = seg*(mean_uv-uv).square()
    return uv_var.mean()

def fg_bg_contrast(model_out,gt):
    attns=model_out["fg_bg_response"]
    def norm(AA):
        AA = AA.view(AA.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        return AA
    fg,bg = [norm(x) for x in attns]
    return (fg*bg).mean()

def slot_contrasitve_dist(model_out,gt):
    sz  = F.normalize(model_out["fg_slot_z"].permute(1,0,2),dim=-1)[:,1:] # b p z
    pairs = torch.combinations(torch.arange(sz.size(1)),2).T
    dist = torch.sub(*sz[:,pairs].permute(1,0,2,3)).abs()
    return (2-dist).mean()

def pose_mag(model_out,gt):
    rot_trans = model_out["cam_vec"]
    return rot_trans.norm(dim=-1).mean()

def coords_cos_contrast(model_out,gt):

    sz = model_out["cam_vec"] # b p z

    pairs = torch.combinations(torch.arange(sz.size(1)),2).T
    sim = (F.cosine_similarity(*sz[:,pairs].permute(1,0,2,3),dim=-1)+1)/2

    return sim.mean()


def slot_contrasitve_cos(model_out,gt):

    sz = model_out["fg_slot_z"].permute(1,0,2) # b p z

    pairs = torch.combinations(torch.arange(sz.size(1)),2).T
    sim = (F.cosine_similarity(*sz[:,pairs].permute(1,0,2,3),dim=-1)+1)/2

    return sim.mean()

"""
from torchvision.models import vgg16
def get_perceptual_net(layer=4):
    print("making perceptual net")
    assert layer > 0
    idx_set = [None, 4, 9, 16, 23, 30]
    idx = idx_set[layer]
    vgg = vgg16(pretrained=True)
    loss_network = nn.Sequential(*list(vgg.features)[:idx]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    return loss_network
from torchvision.transforms import Normalize
vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#net=get_perceptual_net().cuda() #ugly refactor
def perceptual_loss(model_out,gt):
    return F.mse_loss(*[net(vgg_norm(src["rgb"].flatten(0,1).permute(0,2,1)
                            .unflatten(-1,(64,64)))) for src in (model_out,gt)])
"""
def attn_sup(model_out,gt):
    gt_seg = torch.stack(((~gt["seg"].bool()).float(),gt["seg"]))[:,:,:1,:,0]
    gt_seg = F.interpolate(gt_seg.flatten(0,1).squeeze().unflatten(-1,(64,64)
                          ).unsqueeze(1),scale_factor=.5).squeeze().flatten(1,2)
    return F.mse_loss(model_out["slot_attn"].flatten(0,1).squeeze(),gt_seg)
    #return F.mse_loss(model_out["slot_attn"], gt_seg[:,:,:1,:,0].squeeze(-1)#)
# Consistency between slot attention on first query image and conv soft seg
def slot_seg_consistency(model_out,gt):
    return F.mse_loss(model_out["slot_attn"],
                      model_out["soft_seg"][:,:,:,0].squeeze(-1))
# Consistency between conv soft seg and slot attention on [1:] query images
def secondary_seg_consistency(model_out,gt):
    pass

def rgb_loss(model_out,gt):
    return F.mse_loss(model_out["rgb"],gt["rgb"])
def rgb_vid(model_out,gt):
    #return F.mse_loss(model_out["rgb"],torch.stack((gt["rgb"],gt["rot_rgb"])))
    return F.mse_loss(model_out["rgb"],gt["rgb"])
def percept(model_out,gt):
    return loss_fn_alex(*[src.flatten(0,1).permute(0,2,1).unflatten(-1,(64,64))
            for src in [model_out["coarse_rgb"],gt["rgb"]]]).mean()

def alpha_loss(model_out,gt):
    alpha=model_out["alpha"]
    return F.mse_loss(alpha,torch.ones_like(alpha))

# GT pose supervised world2model for testing
def pose_loss(model_out,gt):

    pose_est = model_out["world2model"].squeeze().flatten(0,1)
    pose_gt  = gt["pose2"].squeeze().permute(1,0,2,3).flatten(0,1)

    angle_loss = transforms.so3_relative_angle(pose_est[:,:3,:3],pose_gt[:,:3,:3]).mean()
    translation_loss = F.mse_loss(pose_est[:,:3,-1],pose_gt[:,:3,-1])

    pose_loss = 1e0*(5e0*angle_loss+1e0*translation_loss)
    return pose_loss

# Self supervised analytical depth and regressed depth consistency loss
def analy_depth_loss(model_out,gt):
    return ((model_out["est_depths"]-model_out["analy_depth"]).square()
            *model_out["analy_valid"]).mean()

# Ground truth supervised depth loss
def gt_depth_loss(model_out,gt):
    return ( (model_out["composed_depth"]-gt["depth"]).square()*gt["depth_mask"] ).mean()

# Edge aware depth smoothness loss
def depth_smoothness(model_out,gt):
    im_sl = int(model_out["rgb"].size(-2)**(1/2))
    depth_grad = spatial_gradient(model_out["composed_attn"].squeeze(-1
            ).unflatten(-1, (im_sl,im_sl))).abs().sum(1,keepdim=True)
    rgb_grad = spatial_gradient(gt["rgb"].squeeze().permute(0,2,1
               ).unflatten(-1, (im_sl,im_sl))).abs().sum(1,keepdim=True)
    # test apply seg smoothness too
    seg_grads = [spatial_gradient(model_out["soft_seg"][i].squeeze(-1)
                      .unflatten(-1, (im_sl,im_sl))).abs() for i in range(2)]

    return sum([(depth_grad*torch.exp(-rgb_grad)).sum(2).mean()
                        for src_grad in (depth_grad,seg_grads[0],seg_grads[1])])

    #return (depth_grad*torch.exp(-rgb_grad)).sum(2).mean()

# Ground truth segmentation loss (for testing supervised segmentation capability)
def seg_loss(model_out,gt):
    expanded_gt_seg = torch.cat([(gt["seg"]==id_).unsqueeze(0)
                                        for id_ in gt["seg"].unique()]).float()
    return F.mse_loss(model_out["soft_seg"],expanded_gt_seg)

# Enforces that background phi (assumed to be first element in first(phi) dim)'s
# alpha is 1 everywhere since every unoccluded ray should intersect with it
def bg_alpha_loss(model_out,gt):
    bg_alpha = model_out["alpha"][0]
    return F.mse_loss(bg_alpha,torch.ones_like(bg_alpha))









def image_loss(model_out, gt, mask=None, model=None, val=False):
    gt_rgb = gt['rgb']
    # return nn.L1Loss()(gt_rgb, model_out['rgb']) * 200
    return {'img_loss':nn.MSELoss()(gt_rgb, model_out['rgb']) * 200}, {}

class LFLoss():
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict.update(image_loss(model_out, gt)[0])
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight
        return loss_dict, {}

class CompositeLoss():
    def __init__(self, loss_fns, img_shape=None):
        self.img_shape = img_shape
        self.loss_fns = loss_fns

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_summaries = {}
        for loss_fn in self.loss_fns:
            loss, summ = loss_fn(model_out, gt, model, val)
            loss_dict.update(loss)
            loss_summaries.update(summ)

        return loss_dict, loss_summaries

