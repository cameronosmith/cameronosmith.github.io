import comet_ml

import sys, os
sys.path.append(os.path.join(sys.path[0],'..'))

from pdb import set_trace as pdb

import torch.nn as nn

import models 
import summaries
import loss_functions
import custom_layers

import util

from train import train, rapid_test
testing = len(sys.argv)>1 or False
mode=[train,rapid_test][testing]
use_gan=False
sato = True
sato_cpu=testing and sato
if sato_cpu and not testing: raise Exception("DONT SUBMIT WITH SATO ON")

num_phi=5
run_large=True
print("\n\n USING NUM PHI %d \n\n"%num_phi)

model = models.CLFN(phi_latent=128, phi_out_latent=64,
                hyper_hidden=1,phi_hidden=2,concat_phi=False,
                img_feat_dim=128, sato_cpu=sato_cpu, num_phi=num_phi,zero_bg=False).cuda()

summary_fns = [
                summaries.rgb,
                summaries.seg_vid,
                summaries.slot_attn_vid,
                ]
loss_fns    = [
               (loss_functions.rgb, 25e2,None),
               (loss_functions.latent_penalty, 2e1,None),
               (loss_functions.lpips_loss, 6e0,(3000,2000,100000)),
               ]
mode({
        "data_dir":f"{datasets_path}/fmt_room_chair_train",
        "val_dir":f"{datasets_path}/fmt_room_chair_test",
        "batch_size": 1 if testing else 1,
        "context_is_last": False,
        "pin_mem":False,
        "model":model,
        "val_iters":10,
        "lr":5e-5,
        "update_iter":5000,
        "num_context":1,
        "num_query":1 if testing else 1,
        "num_img":None,
        "num_inst":None,
        "img_sidelength":128,
        "summary_fns":summary_fns,
        "loss_fns":loss_fns,
        })
