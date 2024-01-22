import torch

from utils.utils import safe_normalize, get_rays
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader


def circle_poses(device, radius=1.25, theta=60, phi=0, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    
    return poses

def rand_poses(size, device, radius_range=[4, 4.5], theta_range=[0, 120], phi_range=[0, 360], \
               angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    
    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, thetas, phis, radius

"""
Generate rays from random/default viewpoint
"""
class NeRFDataset:
    def __init__(self, opt, device, type='train', H=64, W=64, size=100, near=2., far=6.):
        self.opt = opt
        self.device = device

        self.type = type
        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.W / 2
        self.cy = self.H / 2

        self.near = near
        self.far = far

    def get_default_view_data(self):
        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx = W / 2
        cy = H / 2

        thetas = torch.FloatTensor([self.opt.default_theta]).to(self.device)
        phis = torch.FloatTensor([self.opt.default_phi]).to(self.device)
        radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
        poses = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, \
                             return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        rays = get_rays(poses, intrinsics, H, W, -1)

        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'near': self.near,
            'far': self.far,
        }

        return data
    
    def collate(self, index):
        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            poses, thetas, phis, radius = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.default_theta]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
 
            # fixed focal
            fov = self.opt.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        
        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'near': self.near,
            'far': self.far,
        }

        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader