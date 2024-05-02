import torch
import MinkowskiEngine as ME
from models.encoder3d import Encoder3D
from models.encoder2d import Encoder2D
from models.unet3d import UNet3D
import pathlib
from typing import List, Tuple, Dict, Union
from PIL import Image
import numpy as np
import torch.nn.functional as F
from utils import *
import pickle
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

models_dict = {
    "enc2d": Encoder2D(in_ch=4, output_dim=16),
    "enc3d": Encoder3D(1, 16, D=3, planes=(32, 48, 64)),
    "unet3d": UNet3D(32, 4**2, f_maps=[32, 48, 64, 80], mode="nearest"),
}

EPS = 1e7


class Eval(object):

    def __init__(self, model_weight_paths: List[pathlib.Path], device):
        print(model_weight_paths)
        self.device = device
        self.models: dict = self.load_models(model_weight_paths)
        # see eval_nyu
        self.opt_res = 16
        self.upscale = 4
        self.z_step = 10 / (self.opt_res - 1)

    def load_models(
        self, model_weight_paths
    ) -> Dict[str, Union[Encoder2D, Encoder3D, UNet3D]]:
        res: Dict[str, Union[Encoder2D, Encoder3D, UNet3D]] = dict()

        def load_model(weights_path: pathlib.Path, model_name: str):
            model = models_dict.get(model_name).to(device=self.device)
            print(weights_path.as_posix())
            model.load_state_dict(
                torch.load(weights_path.as_posix()),
                strict=False,
            )
            model.eval()
            res.update({model_name: model})

        for model_path in model_weight_paths:
            name = (model_path.as_posix().split("/")[-1]).split(".")[0]
            print(f"name - {name}")
            load_model(model_path, name)

        return res

    def fusion(self, sout, feat2d):
        # sparse tensor to dense tensor
        B0, C0, H0, W0 = feat2d.size()
        dense_output_, min_coord, tensor_stride = sout.dense(
            min_coordinate=torch.IntTensor([0, 0, 0])
        )
        dense_output = dense_output_[:, :, : self.opt_res, :H0, :W0]
        B, C, D, H, W = dense_output.size()
        feat3d_ = torch.zeros((B0, C0, self.opt_res, H0, W0), device=feat2d.device)
        feat3d_[:B, :, :D, :H, :W] += dense_output

        # construct type C feat vol
        mask = (torch.sum((feat3d_ != 0), dim=1, keepdim=True) != 0).float()
        mask_ = mask + (
            1 - torch.sum(mask, dim=2, keepdim=True).repeat(1, 1, mask.size(2), 1, 1)
        )
        feat2d_ = feat2d.unsqueeze(2).repeat(1, 1, self.opt_res, 1, 1) * mask_
        return torch.cat([feat2d_, feat3d_], dim=1)

    def depth2MDP(self, dep):
        # Depth to sparse tensor in MDP (multiple-depth-plane)
        idx = torch.round(dep / self.z_step).type(torch.int64)
        idx[idx > (self.opt_res - 1)] = self.opt_res - 1
        idx[idx < 0] = 0
        inv_dep = idx * self.z_step
        res_map = (dep - inv_dep) / self.z_step

        B, C, H, W = dep.size()
        ones = (idx != 0).float()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_ = torch.stack((grid_y, grid_x), 2).to(dep.device)
        # grid_ = self.grid.clone().detach()
        grid_ = grid_.unsqueeze(0).repeat((B, 1, 1, 1))
        points_yx = grid_.reshape(-1, 2)
        point_z = idx.reshape(-1, 1)
        m = (idx != 0).reshape(-1)
        points3d = torch.cat([point_z, points_yx], dim=1)[m]
        split_list = torch.sum(ones, dim=[1, 2, 3], dtype=torch.int).tolist()
        coords = points3d.split(split_list)
        # feat = torch.ones_like(points3d)[:,0].reshape(-1,1)       ## if occ to feat
        feat = res_map
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.size(1))[m]  ## if res to feat

        # Convert to a sparse tensor
        in_field = ME.TensorField(
            features=feat,
            coordinates=ME.utils.batched_coordinates(coords, dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=dep.device,
        )
        return in_field.sparse()

    def upsampling(self, cost, res=64, up_scale=None):
        # if up_scale is None not apply per-plane pixel shuffle
        if not up_scale == None:
            b, c, d, h, w = cost.size()
            cost = cost.transpose(1, 2).reshape(b, -1, h, w)
            cost = F.pixel_shuffle(cost, up_scale)
        else:
            cost = cost.squeeze(1)
        prop = F.softmax(cost, dim=1)
        pred = disparity_regression(prop, res)
        return pred

    def get_sparse_depth(self, dep, num_sample=500):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel * height * width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))
        print(f"mask - {mask}")

        dep_sp = dep * mask.type_as(dep)
        print(f"dep_sp - {dep_sp.size()}")
        return dep_sp

    def eval(self, image: Image, point_cloud: Image):
        sp_depth = self.get_sparse_depth(point_cloud)
        image = image.unsqueeze(0)
        sp_depth = sp_depth.unsqueeze(0)
        print(image.size())
        print(sp_depth.size())
        in_2d = torch.cat([image, sp_depth], 1)
        in_3d = self.depth2MDP(sp_depth)
        print(self.models.keys())
        feat2d = self.models["enc2d"](in_2d)
        feat3d = self.models["enc3d"](in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        ## [step 2] Cost Volume Prediction
        cost_vol = self.models["unet3d"](rgbd_feat_vol)

        ## [step 3] Depth Regression
        pred = (
            self.upsampling(cost_vol, res=self.opt_res, up_scale=self.upscale)
            * self.z_step
        )
        return pred


def get_test_data(image_dir_path: pathlib.Path, lidar_dir_path: pathlib.Path):
    image_path = next(image_dir_path.rglob("*.jpg"))
    image_timestamp = int(image_path.as_posix().split("/")[-1].split(".")[0])
    print(image_timestamp)
    print(image_path)
    for lida_path in lidar_dir_path.rglob("*.npy"):
        lidar_timestamp = int(lida_path.as_posix().split("/")[-1].split(".")[0])
        delta = abs(lidar_timestamp - image_timestamp)
        print(delta)
        if delta < EPS:
            print(image_timestamp)
            print(image_path)
            print(lidar_timestamp)
            print(lida_path)
            return image_path, lida_path

    return None, None


class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)


if __name__ == "__main__":
    test_image_dir_path: pathlib.Path = pathlib.Path(
        "/test_images"
    )
    test_lidar_dir_path: pathlib.Path = pathlib.Path(
        "/lidar_points"
    )
    test_image, lidar_array = get_test_data(test_image_dir_path, test_lidar_dir_path)
    model_dir = pathlib.Path(
        "/depth_completion/CostDCNet/weights"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = np.fromfile(test_image)
    img = Image.fromarray(img, mode="RGB")
    point_array = np.fromfile(lidar_array)
    point_cloud = Image.fromarray(point_array, mode="F")
    t_rgb = T.Compose(
        [
            T.Resize(228),
            T.CenterCrop((228, 304)),
            T.ToTensor(),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    t_dep = T.Compose(
        [T.Resize(228), T.CenterCrop((228, 304)), ToNumpy(), T.ToTensor()]
    )
    # print([el for el in model_dir.rglob("*d.pth")])
    eval_model = Eval([el for el in model_dir.rglob("*d.pth")], device)
    test = eval_model.eval(t_rgb(img).to(device), t_dep(point_cloud).to(device))
    result_img = test.cpu().detach().numpy()
    cv2.imshow("result", result_img.reshape((228, 304)))
    cv2.waitKey(0)