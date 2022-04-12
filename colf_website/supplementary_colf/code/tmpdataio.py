import cv2
import os
import torch
import imageio
import numpy as np
from glob import glob
import data_util
import util
from collections import defaultdict
import skimage.filters
from pdb import set_trace as pdb
from itertools import combinations
from random import choice

class SceneInstanceDataset(torch.utils.data.Dataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,
                 input_img_sidelength=None,
                 img_sidelength=None,
                 num_images=None,
                 use_seg=False,
                 num_seg=-1,
                 use_depth=False,
                 pose2=False,
                 cache=None):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.input_img_sidelength = input_img_sidelength
        self.instance_dir = instance_dir
        self.cache = cache

        self.interp_mode = cv2.INTER_LINEAR

        pose_dir  = os.path.join(instance_dir, "pose")

        color_dir = os.path.join(instance_dir, "rgb")
        #color_dirs= [os.path.join(instance_dir[:-2]+f"_{i}", "rgb") for i in range(3)]
        #flow_dirs = [os.path.join(instance_dir[:-2]+f"_{i}", "flow") for i in range(3)]
        #comp_dirs = [os.path.join(instance_dir[:-2]+f"_{i}", "con_comps") for i in range(3)]

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        #self.color_paths = list(filter(lambda x:"context" in x or "19" in x or "18" in x,sorted(data_util.glob_imgs(color_dir))))
        #self.colors_paths = [sorted(data_util.glob_imgs(c_dir)) for c_dir in color_dirs]
        #self.flows_paths  = [sorted(data_util.glob_ext(c_dir,[".pt"])) for c_dir in flow_dirs]
        #self.comps_paths  = [sorted(data_util.glob_ext(c_dir,[".pt"])) for c_dir in comp_dirs]
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))
        self.instance_name = os.path.basename(os.path.dirname(self.instance_dir))

        if specific_observation_idcs is not None:
            self.color_paths = util.pick(self.color_paths, specific_observation_idcs)
            self.pose_paths  = util.pick(self.pose_paths, specific_observation_idcs)
        elif num_images is not None:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = util.pick(self.color_paths, idcs)
            self.pose_paths  = util.pick(self.pose_paths, idcs)

        dummy_img = data_util.load_rgb(self.color_paths[0])
        self.org_sidelength = dummy_img.shape[1]

        if self.org_sidelength < self.img_sidelength:
            uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32).transpose(1, 2, 0)
            self.intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                             trgt_sidelength=self.img_sidelength)
        else:
            #uv = np.mgrid[0:self.org_sidelength, 0:self.org_sidelength].astype(np.int32).transpose(1, 2, 0)
            uv = np.mgrid[0:self.org_sidelength, 0:self.org_sidelength].astype(np.float).transpose(1, 2, 0)
            uv = cv2.resize(uv, (self.img_sidelength, self.img_sidelength), interpolation=self.interp_mode)
            context_uv = cv2.resize(uv, (64,64), interpolation=self.interp_mode)
            self.intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                             trgt_sidelength=self.org_sidelength)

        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        self.uv = uv.reshape(-1, 2).float()
        context_uv = torch.from_numpy(np.flip(context_uv, axis=-1).copy()).long()
        self.context_uv = context_uv.reshape(-1, 2).float()

        self.intrinsics = torch.Tensor(self.intrinsics).float()

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        #return min(len(self.pose_paths), len(self.color_paths))
        return len(self.color_paths)

    def __getitem__(self, idx, context=False,input_context=True):

        #trot_pair = choice(list(combinations(range(len(self.colors_paths)),2)))
        #base_dir = self.colors_paths[0]
        base_dir = self.color_paths
        #base_dir,rot_dir = [self.colors_paths[i] for i in rot_pair]
        #flow_dir = self.flows_paths[rot_pair[0]]
        #comp_dir = self.comps_paths[rot_pair[0]]

        ### Highway dataset mod
            
        #context_idx=len(self.color_paths)-1
        """
        if not context and idx==context_idx:
            idx = 0
        elif context:
        """
        #if context:
        #    idx=context_idx
        ###

        key = f'{self.instance_idx}_{idx}'
        if (self.cache is not None) and (key in self.cache):
            #rgb, rot_rgb, pose,flow,comp = self.cache[key]
            rgb, pose = self.cache[key]
        else:
            try:
                rgb = data_util.load_rgb(base_dir[idx])
            except:
                print(base_dir,idx)
                pdb()
            pose = data_util.load_pose(self.pose_paths[idx])
            #flow = torch.load(flow_dir[idx])
            #comp = torch.load(comp_dir[idx])
            if (self.cache is not None) and (key not in self.cache):
                #self.cache[key] = rgb, rot_rgb, pose#,flow, comp
                #self.cache[key] = rgb, pose,flow, comp
                self.cache[key] = rgb, pose

        ### Highway dataset mod
        img_sidelength=self.img_sidelength
        #if context and input_context: img_sidelength=256
        if context and input_context: img_sidelength=128
        #if context and input_context: img_sidelength=64
        #if context and input_context: img_sidelength=32
            
        ###
        #rgb = cv2.resize(rgb, (self.img_sidelength, self.img_sidelength), interpolation=self.interp_mode)
        rgb = cv2.resize(rgb, (img_sidelength, img_sidelength), interpolation=self.interp_mode)
        rgb = rgb.reshape(-1, 3)

        #flow = torch.nn.functional.interpolate(flow[None],(self.img_sidelength, 
        #                        self.img_sidelength),mode="bilinear")[0].flatten(1,2).T
        #comp = torch.nn.functional.interpolate(comp[None,None].float(),(self.img_sidelength, 
        #                        self.img_sidelength),mode="nearest")[0].flatten(1,2).T

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
            #"input_rgb": torch.from_numpy(input_rgb if input_rgb is not None else rgb).float(),
            "rgb": torch.from_numpy(rgb).float(),
            "cam2world": torch.from_numpy(pose).float(),
            "imgname":self.color_paths[idx],
            "uv": self.uv,
            "context_uv": self.context_uv,
            "intrinsics": self.intrinsics,
            "height_width": torch.from_numpy(np.array([self.img_sidelength, self.img_sidelength])),
            "instance_name": self.instance_name
        }

        """
        mask_path=self.color_paths[idx].replace("rgb","inst")
        mask = imageio.imread(mask_path)
        mask = skimage.img_as_float32(mask)
        mask = cv2.resize(mask, (self.img_sidelength, self.img_sidelength),
                                             interpolation=cv2.INTER_NEAREST)
        maskattn = cv2.resize(mask, (128,128), interpolation=cv2.INTER_NEAREST)
        maskattn = maskattn.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        sample["gt_mask"]=mask
        sample["gt_maskattn"]=maskattn

        # Dataset optional vars
        for var,name in []:
            if var is not None: sample[name]=var
        """

        return sample


def get_instance_datasets(root, max_num_instances=None, specific_observation_idcs=None,
                          cache=None, sidelen=None, max_observations_per_instance=None):
    instance_dirs = sorted(glob(os.path.join(root, "*/")))
    assert (len(instance_dirs) != 0), f"No objects in the directory {root}"

    if max_num_instances != None:
        instance_dirs = instance_dirs[:max_num_instances]

    all_instances = [SceneInstanceDataset(instance_idx=idx, instance_dir=dir,
                      specific_observation_idcs=specific_observation_idcs, img_sidelength=sidelen,
                                          cache=cache, num_images=max_observations_per_instance)
                     for idx, dir in enumerate(instance_dirs)]
    return all_instances


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 num_context, num_trgt, root_dir,
                 vary_context_number=False,
                 query_sparsity=None,
                 img_sidelength=None,
                 input_img_sidelength=None,
                 max_num_instances=None,
                 max_observations_per_instance=None,
                 specific_observation_idcs=None,
                 test=False,
                 test_context_idcs=None,
                 cache=None,
                 use_seg=False,
                 use_depth=False,
                 pose2=False,
                 num_seg=-1,
                 ):

        self.num_context = num_context
        self.num_trgt = num_trgt
        self.query_sparsity = query_sparsity
        self.img_sidelength = img_sidelength
        self.vary_context_number = vary_context_number
        self.cache = cache
        self.test = test
        self.test_context_idcs = test_context_idcs

        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))
        print(f"Root dir {root_dir}, {len(self.instance_dirs)} instances")

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances is not None:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        #subdirs = set([x.split("/")[-2].split("_")[0]+"_0" for x in self.instance_dirs])
        #subdirs = self.instance_dirs
        #self.instance_dirs = [os.path.join(root_dir,subdir) for subdir in subdirs]
        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   input_img_sidelength=input_img_sidelength,
                                                   num_images=max_observations_per_instance,
                                                   cache=cache,
                                                   use_seg=use_seg,
                                                   use_depth=use_depth,
                                                   num_seg=num_seg,
                                                   pose2=pose2,
                                                   )
                              for idx, dir in enumerate(self.instance_dirs)
                                    ]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def sparsify(self, dict, sparsity):
        new_dict = {}
        if sparsity is None:
            return dict
        else:
            # Sample upper_limit pixel idcs at random.
            rand_idcs = np.random.choice(self.img_sidelength**2, size=sparsity, replace=False)
            for key in ['rgb', 'uv']:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ['rgb', 'uv']:
                    new_dict[key] = dict[key]

            return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        if self.test:
            obj_idx = 0
            while idx >= 0:
                idx -= self.num_per_instance_observations[obj_idx]
                obj_idx += 1
            return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])
        else:
            return np.random.randint(self.num_instances), 0

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx):
        context = []
        trgt = []
        post_input = []

        obj_idx, det_idx = self.get_instance_idx(idx)

        if self.vary_context_number:
            num_context = np.random.randint(1, self.num_context+1)

        if not self.test:
            try:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=False,
                                               size=self.num_context+self.num_trgt)
            except:
                sample_idcs = np.random.choice(len(self.all_instances[obj_idx]), replace=True,
                                               size=self.num_context+self.num_trgt)

        
        sample = self.all_instances[obj_idx].__getitem__(idx,context=True,input_context=True)
        context.append(sample)

        """
        for i in range(self.num_context):
            if self.test:
                sample = self.all_instances[obj_idx][self.test_context_idcs[i]]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i]]
            context.append(sample)

            if self.vary_context_number:
                if i < num_context:
                    context[-1]['mask'] = torch.Tensor([1.])
                else:
                    context[-1]['mask'] = torch.Tensor([0.])
            else:
                context[-1]['mask'] = torch.Tensor([1.])
        """

        print(self.num_trgt)
        for i in range(self.num_trgt):
            # Mod adding context img to query imgs
            """
            if i==0:
                sample = self.all_instances[obj_idx].__getitem__(idx,context=True,input_context=False)
            else:
            """
            if self.test:
                sample = self.all_instances[obj_idx][det_idx]
            else:
                sample = self.all_instances[obj_idx][sample_idcs[i+self.num_context]]

            post_input.append(sample)
            post_input[-1]['mask'] = torch.Tensor([1.])

            sub_sample = self.sparsify(sample, self.query_sparsity)
            trgt.append(sub_sample)

        # trgt.append(context[0])

        post_input = self.collate_fn(post_input)
        trgt = self.collate_fn(trgt)

        out_dict =  {'query': trgt, 'post_input': post_input,'context':None}, trgt
        if self.num_context:
            out_dict[0]["context"] = self.collate_fn(context)
        return out_dict


