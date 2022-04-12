import matplotlib
matplotlib.use('Agg')

import diff_operators
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from einops import repeat

from torch.distributions import Normal, kl_divergence
import util
import torchvision
import os
import time

import flow_vis_torch

from einops import rearrange

from pdb import set_trace as pdb
from torchvision.utils import save_image

def tmp(model_out, gt, writer, iter_):
    seg_vid(model_out, gt, writer, iter_)
    rgb(model_out, gt, writer, iter_)
    slot_attn_vid(model_out, gt, writer, iter_)
    return
    slot_attn_vid(model_out, gt, writer, iter_)
    #optflow(model_out, gt, writer, iter_)
    #connected_comps(model_out, gt, writer, iter_)
    seg_vid(model_out, gt, writer, iter_)
    rgb(model_out, gt, writer, iter_)

def eval_dataset(model,dataset):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            drop_last=True, num_workers=4,)
    lpips_loss_all,ssim_loss_all,psnr_loss_all=0,0,0
    ari_loss_all,ari_fg_loss_all=0,0
    num_pred=0
    num_pred_ari=0
    test_i=10
    for i,(model_input, gt) in enumerate(dataloader):
        model_input,gt = [util.dict_to_gpu(x) for x in
                                              (model_input,gt)]
        print(i)
        #if i==test_i:break
        with torch.no_grad(): model_out = model(model_input,gt)

        pred_rgb,gt_rgb=[src["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(64,64))
                    for src in (model_out,gt)]
        model_segs = model_out["soft_seg"].squeeze().max(0)[1].flatten(0,1)
        gt_segs = torch.stack(gt["gt_mask"],1).flatten(0,1)
        for model_seg,gt_seg_rgb in zip(model_segs,gt_segs):
            bg = (((gt_seg_rgb.unique(dim=1)/2+.5)*1000).round().int()==251).all(1)
            gt_seg=torch.zeros_like(model_seg)
            for unique_i,unique_rgb in enumerate(gt_seg_rgb.unique(dim=0)):
                gt_seg[(gt_seg_rgb==unique_rgb).all(1)]=unique_i
            ari = sklearn.metrics.cluster.adjusted_rand_score(model_seg.cpu(),gt_seg.cpu())
            fg_ari = sklearn.metrics.cluster.adjusted_rand_score(
                                        model_seg.cpu()[~bg],gt_seg.cpu()[~bg])
            ari_loss_all+=ari
            ari_fg_loss_all+=fg_ari
            num_pred_ari+=1
        ssim_loss=piq.ssim(pred_rgb*.5+.5,gt_rgb*.5+.5)
        psnr_loss=piq.psnr(pred_rgb*.5+.5,gt_rgb*.5+.5,1)
        lpips_loss= loss_fn_alex(pred_rgb,gt_rgb).mean()
        ssim_loss_all+=ssim_loss
        lpips_loss_all+=lpips_loss
        psnr_loss_all+=psnr_loss
        num_pred+=1
    print("num ari pred",num_pred_ari)
    print("ari ",ari_loss_all/num_pred_ari)
    print("fg ari ",ari_fg_loss_all/num_pred_ari)
    print("ssim",ssim_loss_all/num_pred)
    print("num pred",num_pred)
    print("ssim",ssim_loss_all/num_pred)
    print("lpips",lpips_loss_all/num_pred)
    print("psnr",psnr_loss_all/num_pred)
    pdb();
    z



def log_dataset(model,val_dataset):


    from torch.utils.data  import DataLoader
    val_dataset.test=True
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=1,)
    for i,(model_input, gt) in enumerate(dataloader):break
    model_input,gt = [util.dict_to_gpu(x) for x in (model_input,gt)]

    try: os.mkdir(outdir)
    except: raise Exception("Dir already exists")
    from torchvision.utils import save_image
    from torch.utils.data  import DataLoader
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=4,)
    colors=torch.tensor([ [0.2510, 0.2510, 0.2510],
                        [0.0000, 0.0000, 1.0000],
                        [0.0000, 1.0000, 0.0000],
                       [1.0000, 0.0000, 0.0000],
                       [1.0000, 1.0000, 0.0000],
                       [1.0000, 0.0000, 1.0000],
                       [0.0000, 1.0000, 1.0000],
                       [.737,.737,.737],
                       [0,0,0]
                       ], device='cuda:0')

    test_i=-1
    fromnames=set()
    for i,(model_input, gt) in enumerate(dataloader):

        if i==test_i:
            break
        # only use input query is az=00
        if "az00" not in gt["imgname"][0][0]:
            print("WRONG")
            z
        fromname=gt["imgname"][0][0].split("/")[-1]
        if fromname in fromnames:
            continue
        else:
            fromnames.add(fromname)

        model_input,gt = [util.dict_to_gpu(x) for x in (model_input,gt)]
        print("%d/%d"%(i,len(val_dataset)))
        with torch.no_grad(): model_out = model(model_input)
        rgbs = model_out["rgb"][0]*.5+.5

        segs=model_out["seg"].squeeze(-1).max(0)[1][0]
        print(fromname)
        for name,pred,seg in zip(gt["imgname"],rgbs,segs):

            name=name[0]

            colorseg=torch.zeros(128**2,3).cuda()
            for col_i in range(model_out["rgbs"].size(0)):
                color=colors[col_i].unsqueeze(0).expand(128**2,-1)
                colorseg += (seg==col_i)[:,None]*color
            save_image(colorseg.T.unflatten(-1,(128,128)),
                os.path.join(outdir,fromname.split(".")[0]+"mask"+
                                    name.split("_")[-1]))

            save_image(pred.T.unflatten(-1,(128,128)),
                os.path.join(outdir,fromname.split(".")[0]+
                                    name.split("_")[-1]))



# Taken from pytorch forum
def plot_grad_flow(named_parameters,writer,iter_):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            max_grads.append(p.grad.abs().max().cpu())
            ave_grads.append(p.grad.abs().mean().cpu())

    fig = plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=.2, lw=1, color="c")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=.5, lw=1, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical",fontsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=max(ave_grads)) # zoom in on the lower gradient regions
    plt.grid(False)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient',
                                            'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    writer.add_figure('Gradient Flow',fig,iter_)
    plt.clf()
    plt.close()

def grad_summary(model, writer, iter_):
    #named_grad = [ (k,v if "phi" not in k else model.phi_params[k[4:]])
    #                        for k,v in model.named_parameters() ]
    named_grad = list(filter(lambda x:"phi" not in x[0] and "disc" not in x[0],
                                                    model.named_parameters()))
    names = [x[0].split(".")[0] for x in named_grad]
    sparse_names =[""]+[x+"_start" if (i==0 or y!=x) else x+"_end" if
                                (i==325-3) else "" for i,(y,x,z) in enumerate(
                                           zip(names,names[1:],names[2:]))]+[""]

    named_params = [(sparse_names[i],named_grad[i][1]) for i in range(len(named_grad))]
    plot_grad_flow(named_params,writer,iter_)

def rgb_vid(model_out, gt, writer, iter_,VAL=""):
    for rgb, suffix in ((gt["rgb"],"GT"),(model_out["rgb"][0],"PRED"),
            *[(rgb,f"SLOT_{i}_PRED") for i,rgb in enumerate(model_out["rgbs"][0])]):

        img = rearrange(rgb,"b q n c -> (b q) c n")*.5+.5

        write_img(img, writer, iter_, VAL+f"A_RGB_{suffix}",nrow=model_out["rgb"][0].size(1),
                normalize=True)
    return
    for rgb, suffix in ((gt["rot_rgb"],"GT"),(model_out["rgb"][1],"PRED"),
            *[(rgb,f"SLOT_{i}_PRED") for i,rgb in enumerate(model_out["rgbs"][1])]):

        img = rearrange(rgb,"b q n c -> (b q) c n")*.5+.5

        write_img(img, writer, iter_, f"B_RGB_{suffix}",nrow=model_out["rgb"][1].size(1),
                normalize=True)

def rgb(model_out, gt, writer, iter_,VAL=""):
    coarse=gt["coarse_out"] if "coarse_out" in gt.keys() else None
    gt=gt["query"]
    fine=model_out
    for pref,src in [ ("_",fine)]+([ ("coarse",coarse), ] if coarse is not None else []):
        for rgb, suffix in ((gt["rgb"],"GT"),(src["rgb"],"PRED"),
                *[(rgb,f"SLOT_{i}_PRED") for i,rgb in enumerate(src["rgbs"])]):

            img = rearrange(rgb,"b q n c -> (b q) c n")*.5+.5

            #write_img(img, writer, iter_, VAL+f"{pref}RGB_{suffix}",nrow=model_out["coarse_rgb"].size(1),
            write_img(img, writer, iter_, VAL+f"{pref}RGB_{suffix}",nrow=model_out["rgb"].size(1),
                    normalize=True)

def slot_attn_vid(model_out, gt, writer, iter_,VAL=""):
    n_phi = model_out["attn"].size(1)
    #for i,attn_name in enumerate(("A_attn","B_attn")):
    # mod for attn sup testing
    """
    if gt["rgb"].size(-1)!=model_out["A_attn"].size(-1):
        attn = F.interpolate(model_out["A_attn"].unflatten(-1,(128,128)),(64,64)).flatten(-2,-1)
    else:
        attn =model_out["A_attn"]
    """
    for pref,attn in ([
                      ("e_",model_out["attn"]),
                     ] + ([("init_fine_",model_out["init_attn"])] if "init_attn" in model_out else [])
                    + ([("coarse",gt["coarse_out"]["attn"])] if "coarse_out" in gt else [])
                     ):
        attn  = attn.permute(1,0,2).unsqueeze(2).permute(0,1,3,2).repeat(1,1,1,3)
        rgb   = gt["context"]["rgb"].flatten(0,1).unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
        applied_attn = attn*rgb
        tmp = torch.stack((attn,applied_attn),1)
        tmp = tmp.permute(2,0,1,3,4).flatten(0,2).permute(0,2,1)
        write_img(tmp, writer, iter_, VAL+pref+"attn", nrow=2*n_phi, normalize=False)

def slot_attn(model_out, gt, writer, iter_):
    n_phi = model_out["slot_attn"].size(0)
    attn  = model_out["slot_attn"].permute(0,1,3,2).repeat(1,1,1,3)
    # tmp mod to use downscale
    #rgb   = gt["input_rgb"][:,0].unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
    rgb   = gt["rgb"][:,0].unsqueeze(0).expand(n_phi,-1,-1,-1)*.5+.5
    # upscale attn to rgb size if needed todo
    applied_attn = attn*rgb
    tmp = torch.stack((attn,applied_attn),1)
    tmp = tmp.permute(2,0,1,3,4).flatten(0,2).permute(0,2,1)
    write_img(tmp, writer, iter_, "Attention(BG,rgb*BG,FG,rgb*FG)",
            nrow=2*n_phi, normalize=False)

def seg_vid(model_out, gt, writer, iter_,VAL=""):
    #coarse=gt["coarse_out"]
    gt=gt["query"]

    for pref,soft_seg,rgb in [
                                #("Coarse",coarse["seg"],coarse["rgbs"]),
                                ("model",model_out["seg"],model_out["rgbs"])
                                ]:

        n_phi = soft_seg.size(0)
        soft_seg = soft_seg.repeat(1,1,1,1,3)
        for i in range(n_phi):
            seg = soft_seg[i].flatten(0,1)
            img=torch.stack((seg,rgb[i].flatten(0,1)*seg),1).flatten(0,1).permute(0,2,1)
            write_img(img, writer, iter_, VAL+f"{pref}_Slot_{i}_seg",nrow=2*soft_seg.size(2))


def seg(model_out, gt, writer, iter_):

    n_phi = model_out["soft_seg"].size(0)
    soft_seg = model_out["soft_seg"].squeeze(2).repeat(1,1,1,1,3)
    for i in range(n_phi):
        seg = soft_seg[i].flatten(0,1)
        img=torch.stack((seg,model_out["rgbs"][i].flatten(0,1)*seg),1
                ).flatten(0,1).permute(0,2,1)
        write_img(img, writer, iter_, f"Slot_{i}_seg",nrow=2*soft_seg.size(2))

def depth(model_out, gt, writer, iter_):

    est_depth = model_out["composed_depth"]
    for src,suffix in ((est_depth,"pred"),(gt["depth"]*gt["depth_mask"],"gt")):
        write_img(src.squeeze(-1), writer, iter_, "depth_"+suffix)
        write_img(1/(1e-5+src).squeeze(-1), writer, iter_, "inv_depth_"+suffix)

def alpha(model_out, gt, writer, iter_):

    img = rearrange(model_out["alpha"],"p 1 b n 1 -> (b p) 1 n")
    write_img(img, writer, iter_, "alpha")

def write_img(imgs,writer,iter_,title,nrow=8,write_local=True,normalize=True):

    img_sl   = int(imgs.size(-1)**(1/2))
    img_grid = torchvision.utils.make_grid(imgs.unflatten(-1,(img_sl,img_sl)),
            scale_each=False, normalize=normalize,nrow=nrow).cpu().detach().numpy()
    if writer is None and write_local:
        plt.imshow(img_grid.transpose(1,2,0))
        plt.axis('off')
        plt.close()
    elif writer is not None:
        writer.add_image(title, img_grid, iter_)


def mpl_plot_pytorch_img(ax, pytorch_img):
    '''pytorch img shape (b, ch, h, w)'''
    img = util.detach_all(pytorch_img[0].permute(1, 2, 0))
    img += 1.
    img /= 2.
    img = np.clip(img, 0, 1)
    ax.imshow(img)

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def detach(tensor):
    return tensor.detach().cpu().numpy()


def multiview_summary(model, model_input, ground_truth, model_output, writer, iter, prefix="", img_shape=None):
    intersections = detach(model_output['intersections'][0])
    ray_dir = detach(model_output['ray_dir'][0])
    new_ray_dir = detach(model_output['new_ray_dirs'][0])
    ray_origin = detach(model_output['ray_origin'][0, 0])

    plt.switch_backend('agg')
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(intersections[..., 0], intersections[..., 1], intersections[..., 2])
    ax.scatter(ray_origin[0], ray_origin[1], ray_origin[2])

    ray_origin = np.tile(ray_origin[None], (ray_dir.shape[0], 1))
    ax.quiver(ray_origin[:, 0], ray_origin[:, 1], ray_origin[:, 2], ray_dir[:, 0], ray_dir[:, 1], ray_dir[:, 2],
              length=0.1)
    ax.quiver(intersections[..., 0], intersections[..., 1], intersections[..., 2],
              -new_ray_dir[..., 0], -new_ray_dir[..., 1], -new_ray_dir[..., 2], length=0.1)
    writer.add_figure(prefix + 'intersections', fig, global_step=iter)


class GeneralImgSummary():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
        for key in self.keys:
            with torch.no_grad():
                depth = util.lin2img(util.flatten_first_two(model_output[key]), image_resolution=img_shape)
                writer.add_image(prefix + key,
                                 torchvision.utils.make_grid(depth, scale_each=False, normalize=True).cpu().numpy(),
                                 iter)
                min_max_summary(prefix + key, depth, writer, iter)


def multiview_summary(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    alpha = util.detach_all(util.lin2img(loss_summaries['alpha'], image_resolution=img_shape))
    depth = util.detach_all(util.lin2img(loss_summaries['depth_1'], image_resolution=img_shape))
    depth *= alpha
    rgb_1 = util.lin2img(loss_summaries['rgb_1'], image_resolution=img_shape)
    rgb_2 = util.lin2img(loss_summaries['rgb_2'], image_resolution=img_shape)

    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(141)
    ax.axis("off")
    ax.imshow(alpha[0,0])

    ax = fig.add_subplot(142)
    ax.axis("off")
    ax.imshow(depth[0,0])

    ax = fig.add_subplot(143)
    ax.axis("off")
    mpl_plot_pytorch_img(ax, rgb_1)

    ax = fig.add_subplot(144)
    ax.axis("off")
    mpl_plot_pytorch_img(ax, rgb_2)
    writer.add_figure(prefix + 'multiview_summary', fig, global_step=iter)


def img_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    predictions = model_output['rgb']
    trgt_imgs = ground_truth['rgb']
    indices = model_input['query']['instance_idx']

    predictions = util.flatten_first_two(predictions)
    trgt_imgs = util.flatten_first_two(trgt_imgs)

    with torch.no_grad():
        if 'context' in model_input and model_input['context']:
            context_images = model_input['context']['rgb'] * model_input['context']['mask'][..., None]
            context_images = util.lin2img(util.flatten_first_two(context_images), image_resolution=img_shape)
            writer.add_image(prefix + "context_images",
                             torchvision.utils.make_grid(context_images, scale_each=False, normalize=True).cpu().numpy(),
                             iter)

        output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
        output_vs_gt = util.lin2img(output_vs_gt, image_resolution=img_shape)
        writer.add_image(prefix + "output_vs_gt",
                         torchvision.utils.make_grid(output_vs_gt, scale_each=False,
                                                     normalize=True).cpu().detach().numpy(),
                         iter)

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        writer.add_scalar(prefix + "idx_min", indices.min(), iter)
        writer.add_scalar(prefix + "idx_max", indices.max(), iter)


def slot_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    rgb_slots = model_output['rgb_slots'] # (b*nqry, n_slots, -1, 3)
    attn = model_output['attn']

    rgb_slots = util.lin2img(rgb_slots[0])
    writer.add_image(prefix + "rgb_slots",
                     torchvision.utils.make_grid(rgb_slots, scale_each=False,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)

    attn_masks = util.lin2img(attn[0].unsqueeze(-1))
    writer.add_image(prefix + "attn_masks",
                     torchvision.utils.make_grid(attn_masks, scale_each=False,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)


def intersection_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    # Plot an example ray-sphere intersection
    intersections = torch.stack((model_output['intsec_1'], model_output['intsec_2']), dim=1)[0].detach().cpu().numpy()
    ray_dir = model_output['ray_dir'][0].detach().cpu().numpy()
    ray_origin = model_output['ray_origin'][0, 0].detach().cpu().numpy()

    plt.switch_backend('agg')
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(intersections[1, ..., 0], intersections[1, ..., 1], intersections[1, ..., 2])
    ax.scatter(intersections[0, ..., 0], intersections[0, ..., 1], intersections[0, ..., 2])
    ax.scatter(ray_origin[0], ray_origin[1], ray_origin[2])

    ray_origin = np.tile(ray_origin[None], (ray_dir.shape[0], 1))
    ax.quiver(ray_origin[:, 0], ray_origin[:, 1], ray_origin[:, 2], ray_dir[:, 0], ray_dir[:, 1], ray_dir[:, 2],
              length=0.1)
    writer.add_figure(prefix + 'intersections', fig, global_step=iter)

    # Plot an epipolar image by fixing the back intersections and rotating the front intersections.
    back_intsec = model_output['intsec_1']
    front_intsec = model_output['intsec_2']

    # Slice
    front_intsec = util.lin2img(front_intsec, image_resolution=img_shape)
    center_row = front_intsec.shape[2] // 2
    front_intsec = front_intsec[:1, :, center_row:center_row + 1, :]
    front_intsec = torch.flatten(front_intsec.permute(0, 2, 3, 1), start_dim=1, end_dim=2)

    back_intsec = util.lin2img(back_intsec, image_resolution=img_shape)[:1, :, center_row:center_row + 1, :]
    back_intsec = torch.flatten(back_intsec.permute(0, 2, 3, 1), start_dim=1, end_dim=2)

    slices = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(front_intsec[..., 0].detach().cpu().numpy(),
               front_intsec[..., 1].detach().cpu().numpy(),
               front_intsec[..., 2].detach().cpu().numpy())
    vec_1 = back_intsec[0, 0] - front_intsec[0, 0]
    vec_2 = front_intsec[0, -1] - front_intsec[0, 0]
    axis = F.normalize(torch.cross(vec_1, vec_2), dim=-1).detach().cpu().numpy()

    ax.quiver(0., 0., 0., axis[0], axis[1], axis[2], length=0.3)

    max_angle = 20
    for angle in np.linspace(max_angle * -np.pi / 180, max_angle * np.pi / 180., 100):
        rotvec = axis * angle
        r = torch.from_numpy(R.from_rotvec(rotvec).as_matrix()).to(back_intsec.device)[None, ...].float()

        rot_back = torch.einsum('bkj,bij->bik', r, back_intsec)
        ax.scatter(rot_back[..., 0].detach().cpu().numpy(),
                   rot_back[..., 1].detach().cpu().numpy(),
                   rot_back[..., 2].detach().cpu().numpy())
        light_field_coords = torch.cat((rot_back, front_intsec), dim=-1)

        slices.append(model.sample_light_field(light_field_coords, z=None)['rgb'].reshape(1, -1, 3))
        # slices.append(model.sample_light_field(light_field_coords, model_output['z'][:1])['rgb'].reshape(1, -1, 3))

    writer.add_figure(prefix + 'slice_intersections', fig, global_step=iter)

    # Visualize the line through the image
    source_rgb = util.lin2img(model_output['rgb'][:1, 0].clone().detach(), image_resolution=img_shape)
    source_rgb[:1, :, center_row:center_row + 1, :] = 0.
    writer.add_image(prefix + "slice_rgb",
                     torchvision.utils.make_grid(source_rgb, scale_each=False,
                                                 normalize=True).cpu().detach().numpy(),
                     iter)

    slices = torch.stack(slices, dim=-1)
    slices = slices.permute(0, 2, -1, 1)
    writer.add_image(prefix + "slices",
                     torchvision.utils.make_grid(slices, scale_each=False, normalize=True).detach().cpu().numpy(), iter)


def plucker_slice(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):

    # Gradients of the slice
    with torch.enable_grad():
        lf_fn = model.get_light_field_function()

        # Canonical plucker slice
        # slice_out = util.canonical_plucker_slice(ground_truth['cam2world'][:1], lf_fn, sl=128)
        slice_out = util.get_random_slices(lf_fn, sl=128)

        slice = slice_out['slice']
        print(slice.shape)
        coords = slice_out['st']

        grads = diff_operators.gradient(slice, coords)

    slice = slice.view(-1, 128, 128, 3)
    coords = coords.view(-1, 128, 128, 2)
    grads = grads.view(-1, 128, 128, 2)

    # slice_ft = torch.fft.rfft2(slice, dim=(1, 2))

    confidence = grads.norm(dim=-1, keepdim=True)
    grads, slice, coords, confidence = map(util.detach_all, [grads, slice, coords, confidence])

    slice += 1
    slice /= 2.

    coords = coords[0, ::5, ::5].reshape(-1, 2)
    grads = grads[0, ::5, ::5].reshape(-1, 2)

    # coords[..., 0] *= -1
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(slice[0].transpose(1, 0, -1), extent=[-1, 1, -1, 1], origin='lower')
    ax.quiver(coords[..., 0], coords[..., 1], grads[..., 0], grads[..., 1])

    ax = fig.add_subplot(122)
    ax.imshow((confidence[0]>10).transpose(1, 0, -1), extent=[-1, 1, -1, 1], origin='lower')
    writer.add_figure(prefix + 'slice', fig, global_step=iter)

    # print(slice_ft.shape)
    # fft_grid = torchvision.utils.make_grid(torch.abs(slice_ft).permute(0,-1,1,2), scale_each=True, normalize=True)
    # writer.add_image(prefix + "slice_fft", util.detach_all(fft_grid), iter)


def min_max_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    for k, v in model_output.items():
        if isinstance(v, list):
            v_ = torch.stack(v)
        else:
            v_ = v
        writer.add_scalar(prefix + 'output_'+ k + "_max", v_.max(), global_step=iter)
        writer.add_scalar(prefix + 'output_'+ k + "_min", v_.min(), global_step=iter)

    for k, v in model_input.items():
        if isinstance(v, list):
            v_ = torch.stack(v)
        else:
            v_ = v
        writer.add_scalar(prefix + 'input_'+ k + "_max", v_.max(), global_step=iter)
        writer.add_scalar(prefix + 'input_'+ k + "_min", v_.min(), global_step=iter)

    for k, v in ground_truth.items():
        if isinstance(v, list):
            v_ = torch.stack(v)
        else:
            v_ = v
        writer.add_scalar(prefix + k + "_max", v_.max(), global_step=iter)
        writer.add_scalar(prefix + k + "_min", v_.min(), global_step=iter)


def get_loss_fig(tensor):
    tensor = tensor[0].squeeze(dim=-1)
    img = util.detach_all(util.lin2img(tensor)).transpose(1, 2, 0)
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    fig, ax = plt.subplots()
    ax.imshow(img)
    return fig


def scene(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=None):
    with torch.no_grad():
        scene = detach(util.lin2img(model_output['scene']))
        writer.add_image(prefix + "depth", scene, iter)


class LFSummaries():
    def __init__(self, summary_fns, img_shape=None):
        self.img_shape = img_shape
        self.summary_fns = summary_fns

    def __call__(self, model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix=""):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        for summary_fn in self.summary_fns:
            summary_fn(model, model_input, ground_truth, loss_summaries, model_output,
                       writer, iter, prefix=prefix, img_shape=self.img_shape)
