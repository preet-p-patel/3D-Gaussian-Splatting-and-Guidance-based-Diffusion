import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

SH_C0 = 0.28209479177387814
CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 5.0, 7.0

class TruckDataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}")

        self.masks = []
        self.points = []
        self.images = []
        self.cameras = []

        imgs_root = os.path.join(root, "imgs")
        poses_root = os.path.join(root, "poses")
        points_root = os.path.join(root, "points")
        self.points_path = os.path.join(points_root, "points_10000.npy")

        data_img_size = None
        num_files = len(os.listdir(imgs_root))
        test_idxs = np.linspace(0, num_files, 7).astype(np.int32)[1:-1]
        test_idxs_set = set(test_idxs.tolist())
        train_idxs = [i for i in range(num_files) if i not in test_idxs_set]
        idxs = train_idxs if self.split == "train" else test_idxs

        for i in idxs:
            img_path = os.path.join(imgs_root, f"frame{i+1:06d}.jpg")
            npy_path = os.path.join(poses_root, f"frame{i+1:06d}.npy")

            img_ = imageio.v3.imread(img_path).astype(np.float32) / 255.0

            mask = None
            if img_.shape[-1] == 3:
                img = torch.tensor(img_)  # (H, W, 3)
            else:
                img = torch.tensor(img_[..., :3])  # (H, W, 3)
                mask = torch.tensor(img_[..., 3:4])  # (H, W, 1)
                
            img_size = img.shape[:2]
            h, w = img_size
            
            # Checking if all data samples have the same image size
            if data_img_size is None:
                data_img_size = (w,h) 
            else:
                if data_img_size[0] != img_size[1] or data_img_size[1] != img_size[0]:
                    raise RuntimeError

            pose = np.load(npy_path)
            R, T, F, C = pose[:9].reshape((3,3)), pose[9:12], pose[12:14], pose[14:16]
            
            # Screen space camera
            F = F * min(img_size) / 2 
            C = w / 2 - C[0] * min(img_size) / 2, h / 2 - C[1] * min(img_size) / 2  

            camera = PerspectiveCameras(
                focal_length=torch.tensor(F, dtype=torch.float)[None], 
                principal_point=torch.tensor(C, dtype=torch.float)[None],
                R=torch.tensor(R, dtype=torch.float)[None], 
                T=torch.tensor(T, dtype=torch.float)[None],
                in_ndc=False,
                image_size=((h,w),)
            )

            self.images.append(img)
            self.cameras.append(camera)
            if mask is not None:
                self.masks.append(mask)

        self.img_size = data_img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        masks = None
        if len(self.masks) > 0:
            masks = self.masks[idx]
        return self.images[idx], self.cameras[idx], masks

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch], dim=0)
        cameras = [x[1] for x in batch]

        masks = [x[2] for x in batch if x[2] is not None]
        if len(masks) == 0:
            masks = None
        else:
            masks = torch.stack(masks, dim=0)

        return images, cameras, masks


def colour_depth_q1_render(depth):
    normalized_depth = (depth - CMAP_MIN_NORM) / (CMAP_MAX_NORM - CMAP_MIN_NORM + 1e-8)
    coloured_depth = CMAP_JET(normalized_depth)[:, :, :3]  # (H, W, 3)
    coloured_depth = (np.clip(coloured_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    return coloured_depth

def visualize_renders(scene, gt_viz_img, cameras, img_size):

    imgs = []
    viz_size = (256, 256)
    with torch.no_grad():
        for cam in cameras:
            pred_img, _, _ = scene.render(
                cam, img_size=img_size,
                bg_colour=(0.0, 0.0, 0.0),
                per_splat=-1,
            )
            img = torch.clamp(pred_img, 0.0, 1.0) * 255.0
            img = img.clone().detach().cpu().numpy().astype(np.uint8)

            if img_size[0] != viz_size[0] or img_size[1] != viz_size[1]:
                img = np.array(Image.fromarray(img).resize(viz_size))

            imgs.append(img)

    pred_viz_img = np.concatenate(imgs, axis=1)
    viz_frame = np.concatenate((pred_viz_img, gt_viz_img), axis=0)
    return viz_frame

def load_gaussians_from_ply(path):
    # Modified from https://github.com/thomasantony/splat
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_colours
    }
    return output

def colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs):
    """
    [Q 1.3.1] Computes view-dependent colour given spherical harmonic coefficients
    and direction vectors for each gaussian.

    Args:
        spherical_harmonics     :   A torch.Tensor of shape (N, 48) representing the
                                    spherical harmonic coefficients.
        gaussian_dirs           :   A torch.Tensor of shape (N, 3) representing the
                                    direction vectors pointing from the camera origin
                                    to each Gaussian.

    Returns:
        colours                 :   A torch.Tensor of shape (N, 3) representing the view dependent
                                    RGB colour.
    """
    ### YOUR CODE HERE ###
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2_0 = 1.0925484305920792
    SH_C2_1 = -1.0925484305920792
    SH_C2_2 = 0.31539156525252005
    SH_C2_3 = -1.0925484305920792
    SH_C2_4 = 0.5462742152960396
    SH_C3_0 = -0.5900435899266435
    SH_C3_1 = 2.890611442640554
    SH_C3_2 = -0.4570457994644658
    SH_C3_3 = 0.3731763325901154
    SH_C3_4 = -0.4570457994644658
    SH_C3_5 = 1.445305721320277
    SH_C3_6 = -0.5900435899266435

    c0 = spherical_harmonics[:, :3]  
    colours = SH_C0 * c0

    c1 = spherical_harmonics[:, 3:6]
    c2 = spherical_harmonics[:, 6:9]
    c3 = spherical_harmonics[:, 9:12]
    x, y, z = gaussian_dirs[:, 0], gaussian_dirs[:, 1], gaussian_dirs[:, 2]
    colours = colours - SH_C1 * y[:, None] * c1 + SH_C1 * z[:, None] * c2 - SH_C1 * x[:, None] * c3

    if spherical_harmonics.shape[1] > 12:
        c4 = spherical_harmonics[:, 12:15]
        c5 = spherical_harmonics[:, 15:18]
        c6 = spherical_harmonics[:, 18:21]
        c7 = spherical_harmonics[:, 21:24]
        c8 = spherical_harmonics[:, 24:27]
        xx, yy, zz = x**2, y**2, z**2
        xy, yz, xz = x*y, y*z, x*z
        colours = colours + SH_C2_0 * xy[:, None] * c4 + \
                SH_C2_1 * yz[:, None] * c5 + \
                SH_C2_2 * (2.0 * zz - xx - yy)[:, None] * c6 + \
                SH_C2_3 * xz[:, None] * c7 + \
                SH_C2_4 * (xx - yy)[:, None] * c8
        
    if spherical_harmonics.shape[1] > 27:
        c9 = spherical_harmonics[:, 27:30]
        c10 = spherical_harmonics[:, 30:33]
        c11 = spherical_harmonics[:, 33:36]
        c12 = spherical_harmonics[:, 36:39]
        c13 = spherical_harmonics[:, 39:42]
        c14 = spherical_harmonics[:, 42:45]
        c15 = spherical_harmonics[:, 45:48]
        colours = colours + \
                SH_C3_0 * y[:, None] * (3.0 * xx - yy)[:, None] * c9 + \
                SH_C3_1 * xy[:, None] * z[:, None] * c10 + \
                SH_C3_2 * y[:, None] * (4.0 * zz - xx - yy)[:, None] * c11 + \
                SH_C3_3 * z[:, None] * (2.0 * zz - 3.0 * xx - 3.0 * yy)[:, None] * c12 + \
                SH_C3_4 * x[:, None] * (4.0 * zz - xx - yy)[:, None] * c13 + \
                SH_C3_5 * z[:, None] * (xx - yy)[:, None] * c14 + \
                SH_C3_6 * x[:, None] * (xx - 3.0 * yy)[:, None] * c15
    
    colours = colours + 0.5
    colours = torch.clamp(colours, min=0.0, max=1.0)

    return colours
