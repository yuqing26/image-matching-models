

from matching.base_matcher import BaseMatcher
from matching import WEIGHTS_DIR
import torch
import sys
from pathlib import Path
import gdown
import urllib
from PIL import Image
import numpy as np
from kornia import tensor_to_image
from kornia.color import rgb_to_grayscale

BASE_PATH = Path(__file__).parent.parent.joinpath('third_party/gim')
sys.path.append(str(BASE_PATH))

class GIM_DKM(BaseMatcher):
    from dkm.models.model_zoo.DKMv3 import DKMv3

    weights_src = 'https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view'
    
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.ckpt_path = WEIGHTS_DIR / 'gim_dkm_100h.ckpt'
        
        self.model = DKMv3(weights=None, h=672, w=896)   
        
        self.download_weights()
        self.load_weights()
        
        self.model = self.model.eval().to(device)
        
        
    def download_weights(self):
        if not self.ckpt_path.exists():
            print(f"Downloading {self.ckpt_path.name}")
            gdown.download(GIM_DKM.weights_src,
                                output=str(self.ckpt_path),
                                fuzzy=True)
    def load_weights(self):
        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        self.model.load_state_dict(state_dict)

    def preprocess(self, img):
        # this version of DKM requires PIL images as input
        return Image.fromarray(np.uint8(255*tensor_to_image(img)))       
        
    def _forward(self, img0, img1):
        height0, width0 = img0.shape[-2:]
        height1, width1 = img1.shape[-2:]

        img0 = self.preprocess(img0) # now as PIL img
        img1 = self.preprocess(img1) # now as PIL img
        dense_matches, dense_certainty = self.model.match(img0, img1, device=self.device)
        torch.cuda.empty_cache()
        # sample matching keypoints from dense warp
        sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)
        torch.cuda.empty_cache()
        mkpts0 = sparse_matches[:, :2]
        mkpts1 = sparse_matches[:, 2:]

        # convert to px coords
        mkpts0 = torch.stack((width0 * (mkpts0[:, 0] + 1) / 2, height0 * (mkpts0[:, 1] + 1) / 2), dim=-1,)
        mkpts1 = torch.stack((width1 * (mkpts1[:, 0] + 1) / 2, height1 * (mkpts1[:, 1] + 1) / 2), dim=-1,)
        
        # b_ids = torch.where(mconf[None])[0]
        
        return self.process_matches(mkpts0, mkpts1)
    
class GIM_LG(BaseMatcher):

    weights_src = 'https://github.com/xuelunshen/gim/blob/main/weights/gim_lightglue_100h.ckpt'
    
    def __init__(self, device="cpu", max_keypoints=2048):
        super().__init__(device)
        from gluefactory.superpoint import SuperPoint
        from gluefactory.models.matchers.lightglue import LightGlue

        self.ckpt_path = BASE_PATH / 'weights' / 'gim_lightglue_100h.ckpt'
        
        self.detector = SuperPoint({
            'max_num_keypoints': max_keypoints,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
        
        self.model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })
        
        self.download_weights()
        self.load_weights()

        
    def download_weights(self):
        if not self.ckpt_path.exists():
            print(f"Downloading {self.ckpt_path.name}")
            urllib.request.urlretrieve(GIM_LG.weights_src, self.ckpt_path) 

    def load_weights(self):
        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        self.detector.load_state_dict(state_dict)

        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        
        self.detector = self.detector.eval().to(self.device)
        self.model = self.model.eval().to(self.device)
        
    def preprocess(self, img):
        # convert to grayscale
        return rgb_to_grayscale(img.unsqueeze(0))
        
    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img0 = self.preprocess(img1)

        data = dict(image0=img0, image1=img1)

        scale0 = torch.tensor([1.0, 1.0]).to(self.device)[None]
        scale1 = torch.tensor([1.0, 1.0]).to(self.device)[None]

        size0 = torch.tensor(data["image0"].shape[-2:][::-1])[None]
        size1 = torch.tensor(data["image0"].shape[-2:][::-1])[None]

        data.update(dict(size0=size0, size1=size1))
        data.update(dict(scale0=scale0, scale1=scale1))
        
        pred = {}
        pred.update({k + '0': v for k, v in self.detector({
                                            "image": data["image0"],
                                            "image_size": data["size0"],
                                            }).items()})
        pred.update({k + '1': v for k, v in self.detector({
                                            "image": data["image0"],
                                            "image_size": data["size1"],
                                        }).items()})
        
        pred.update(self.model({**pred, **data,
                           **{'resize0': data['size0'], 'resize1': data['size1']}}))

        kpts0 = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        kpts1 = torch.cat([kp * s for kp, s in zip(pred['keypoints1'], data['scale1'][:, None])])
        
        m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = pred['matches']
        bs = data['image0'].size(0)

        mkpts0 = torch.cat([kpts0[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mkpts1 = torch.cat([kpts1[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
        # b_ids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        # mconf = torch.cat(pred['scores'])

        return self.process_matches(mkpts0, mkpts1)