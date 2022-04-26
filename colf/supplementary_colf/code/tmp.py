"""
This file just defines an abstracted infinite training loop
Ignore the rapid test loop if not testing
"""
import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))
from importlib import reload
from functools import partial
from copy      import deepcopy
import traceback

from dataio import SceneClassDataset
from torch.utils.data import DataLoader

from pdb import set_trace as pdb

# Load data
dataset = SceneClassDataset(1, 4, "storage/datasets/highway", img_sidelength=64)
dataset[0]

dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                        drop_last=True, num_workers=3,)

for model_input, ground_truth in dataloader:
    break
pdb()
pSNARuF

# Fine encoding
        """
        encoder_params = self.hyper_fine_encoder(slots_A.flatten(0,1).detach())
        posenc=positionalencoding2d(self.num_pos_enc,imsize,imsize
                                        ).flatten(1,2).T[None].expand(b,-1,-1).cuda()
        fine_context=torch.cat((posenc,imfeats.detach()),-1)
        masked_img = attn_A[...,None].detach()*fine_context.unsqueeze(1).expand(-1,self.num_phi,-1,-1)
        res_encodings=self.fine_encoder(masked_img.flatten(0,1),params=encoder_params).sum(-2)
        fine_slots = slots_A.detach() + res_encodings.unflatten(0,(b,self.num_phi))
        fine_slots = slots_A.detach()
        """


        #fg_seg = self.compositor(coarse_feats[1:])
        #fg_alpha = self.feat_to_alpha(coarse_feats[1:]).sigmoid().max(0)[0][None]
        #coarse_seg=torch.cat((1-fg_alpha,fg_alpha*fg_seg))
        #coarse_rgb  = (coarse_rgbs[1:]*fg_seg).sum(0)*fg_alpha[0]+coarse_rgbs[0]*(1-fg_alpha[0])
