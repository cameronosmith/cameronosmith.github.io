import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import math

from collections import OrderedDict

try:
    import torch_geometric
    from torch_geometric.nn import Sequential, GraphConv, DenseGraphConv
except:
    print("Graph conv. compositing not available as torch geometric not installed")

from einops import repeat,rearrange
from einops.layers.torch import Rearrange

from pdb import set_trace as pdb #debugging

from scipy.spatial.transform import Rotation as R
from copy import deepcopy

import torchvision
import util

import custom_layers
import geometry
import hyperlayers

import conv_modules

from torchgeometry import rtvec_to_pose

from torch.nn.functional import normalize


# SlotLFN but learning from video
class CLFN(nn.Module):

    def __init__(self,phi_latent=128, num_phi=1, phi_out_latent=64, 
                    no_bg=False,
                    hyper_hidden=1,phi_hidden=2,concat_phi=False,
                    img_feat_dim=128,sato_cpu=False,
                    static_bg=False,zero_bg=False):
        super().__init__()

        if static_bg:
            self.static_bg = custom_layers.FCBlock(
                                    hidden_ch=256,
                                    num_hidden_layers=6,
                                    in_features=6,
                                    out_features=64,
                                    outermost_linear=True,)
            self.static_bg_pix_gen = nn.Sequential( custom_layers.FCBlock(
                            hidden_ch=128, num_hidden_layers=3, in_features=64,
                            out_features=3, outermost_linear=True,
                            norm='layernorm_na'), nn.Tanh() )
        else:
            self.static_bg=None

        self.coarse=True
        self.freeze_coarse_enc=False
        self.zero_bg=zero_bg
        self.concat_phi=concat_phi
        self.no_bg=no_bg

        self.num_phi=num_phi

        num_hidden_units_phi = 256

        self.sato_cpu = sato_cpu
        self.sato_wrap = (lambda mod,x: mod.cpu()(x.cpu()).cuda())\
                                 if sato_cpu else (lambda mod,x:mod(x))

        self.phi = custom_layers.FCBlock(
                                hidden_ch=num_hidden_units_phi,
                                num_hidden_layers=phi_hidden,
                                in_features=6,
                                out_features=phi_out_latent,
                                outermost_linear=True,)
        self.hyper_fg = hyperlayers.HyperNetwork(
                              hyper_in_features=phi_latent,
                              hyper_hidden_layers=hyper_hidden,
                              hyper_hidden_features=num_hidden_units_phi,
                              hypo_module=self.phi)
        self.hyper_bg = hyperlayers.HyperNetwork(
                              hyper_in_features=phi_latent,
                              hyper_hidden_layers=hyper_hidden,
                              hyper_hidden_features=num_hidden_units_phi,
                              hypo_module=self.phi)

        # Maps pixels to features for SlotAttention
        self.img_encoder = nn.Sequential(
                conv_modules.UnetEncoder(bottom=True,z_dim=img_feat_dim),
                Rearrange("b c x y -> b (x y) c")
        )
        if self.no_bg:
            self.slot_encoder = custom_layers.SlotAttentionFG(self.num_phi,
                                                        learned_emb=True,#rebut mod
                                                       in_dim=img_feat_dim,
                                                       slot_dim=phi_latent,)
            print("Slot attn FG")
        else:
            self.slot_encoder = custom_layers.SlotAttention(self.num_phi,
                                                       in_dim=img_feat_dim,
                                                       fg_slot_dim=phi_latent,
                                                       bg_slot_dim=phi_latent,
                                                       max_slot_dim=phi_latent)

        self.feat_to_depth  = custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=1, outermost_linear=True,
                        norm='layernorm_na')
        self.depth_spreader = custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=2,
                        out_features=1, outermost_linear=True,
                        norm='layernorm_na')
        # Maps features to rgb
        self.pix_gen_bg = nn.Sequential( custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=3, outermost_linear=True,
                        norm='layernorm_na'), nn.Tanh() )
        self.pix_gen_fg = nn.Sequential( custom_layers.FCBlock(
                        hidden_ch=128, num_hidden_layers=3, in_features=phi_out_latent,
                        out_features=3, outermost_linear=True,
                        norm='layernorm_na'), nn.Tanh() )
        print(self)

    def compositor(self,feats):
        depth = self.feat_to_depth(feats)
        nodes = rearrange(depth,"p b q pix 1 -> (b q pix) p 1")
        min_depth = nodes.min(1,keepdim=True)[0].expand(-1,feats.size(0),1)
        attn = rearrange(self.depth_spreader(torch.cat((min_depth-nodes,nodes),-1)),
                            "(b q pix) p 1 -> p b q pix 1",
                            p=feats.size(0),b=feats.size(1),q=feats.size(2))
        return attn.softmax(0)+1e-9

    def forward(self,input):

        # Unpack input

        query = input['query']
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]
        cam2world, query_intrinsics, query_uv = util.get_query_cam(input)
        phi_intrinsics,phi_uv = [x.unsqueeze(0).expand(self.num_phi-(0 if self.no_bg else 1),-1,-1,-1)
                                        for x in (query_intrinsics,query_uv)]
        imsize = int(input["context"]["rgb"].size(-2)**(1/2))

        num_context=input["context"]["rgb"].size(1)
        if self.static_bg is not None: self.pix_gen_bg=self.static_bg_pix_gen

        context_rgb = input["context"]["rgb"][:,context_i].permute(0,2,1).unflatten(-1,(imsize,imsize))
        context_cam = input["context"]["cam2world"][:,context_i]
        world2contextcam = repeat(context_cam.inverse(),"b x y -> (b q) x y",q=n_qry)
        fgcam = repeat(world2contextcam @ cam2world,"bq x y -> p bq x y",p=self.num_phi-(0 if self.no_bg else 1))
        fg_coords=geometry.plucker_embedding(fgcam,phi_uv,phi_intrinsics).flatten(0,1)
        bg_coords = geometry.plucker_embedding(cam2world,query_uv,query_intrinsics)

        # Create fg images: img_encoding -> slot attn -> compositor,rgb

        imfeats = self.img_encoder.cuda()(context_rgb.cuda())
        slots, attn_= self.slot_encoder(imfeats) # b phi l 

        # Create phi
        fg_params = self.hyper_fg(repeat(slots[:,(0 if self.no_bg else 1):],"b p l -> p (b q) l",q=n_qry))
        fg_feats = self.phi(fg_coords,params=fg_params)
        if context_i==0: # Only use BG from first context view
            if self.static_bg is None:
                bgslots=int(not self.zero_bg)*slots[:,:1]
                bg_params = self.hyper_bg(repeat( bgslots,"b p l -> p (b q) l",q=n_qry))
                bg_feats = self.phi(bg_coords,params=bg_params)
            else:
                bg_feats = self.static_bg(bg_coords)
            phi_feats = torch.cat((bg_feats,fg_feats))
            phi_feats = rearrange(phi_feats, "(p b q) pix l -> p b q pix l",
                                                    p=self.num_phi,b=b,q=n_qry)
            attn = attn_
        else:
            fg_feats = rearrange(fg_feats, "(p b q) pix l -> p b q pix l",
                                                    p=self.num_phi-1,b=b,q=n_qry)
            phi_feats = torch.cat((phi_feats,fg_feats))
            attn=torch.cat((attn,attn_))
        # Composite over phi dimension to yield single scene
        seg = self.compositor(phi_feats)
        rgbs = torch.cat((self.pix_gen_bg(phi_feats[:1]),
                          self.pix_gen_fg(phi_feats[1:])))
        rgb = (rgbs*seg).sum(0) # b q pix 3

        # Packup for loss fns and summaries
        out_dict = {
            "rgbs":rgbs,
            "rgb": rgb,
            "seg": seg,
            "attn":attn,
            "fg_latent":slots[:,1:],
        }

        return out_dict
