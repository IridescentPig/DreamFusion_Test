import torch
import argparse
import sys

from dataset.nerf_dataset import NeRFDataset
from nerf.nerf import NeRF
from nerf.utils import load_ckpt, Embedding
from utils.utils import seed_everything
from manopth.manolayer import ManoLayer
from utils.trainer import Trainer
import numpy as np
from diffusion.sd_utils import StableDiffusion

import torch.optim as optim


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='deepfloyd-if', help='score model')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidancescale")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--warmup_iters', type=int, default=2000, help="training iters that only use albedo shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1, help="clip grad of rgb space grad to this limit, negative value disables it")
    # network backbone
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    # parser.add_argument('--if_version', type=str, default='1.0', choices=['1.0'], help="IF version")
    # parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--known_view_scale', type=float, default=1., help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3, help="random camera noise added to rays_o and rays_d")

    ### dataset options
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[4.0, 4.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 80], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=4, help="radius for the default view")
    parser.add_argument('--default_theta', type=float, default=90, help="radius for the default view")
    parser.add_argument('--default_phi', type=float, default=0, help="radius for the default view")
    parser.add_argument('--default_fovy', type=float, default=60, help="fovy for the default view")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=10, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=5, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=0.1, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")

    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ncomps = 45
    N_samples = 64
    N_importance = 64

    hand_embedding_xyz = Embedding(3, 10)
    hand_embedding_dir = Embedding(3, 4)
    object_embedding_xyz = Embedding(3, 10)
    object_embedding_dir = Embedding(3, 4)

    hand_nerf_coarse = NeRF()
    hand_nerf_fine = NeRF()
    object_nerf_coarse = NeRF()
    object_nerf_fine = NeRF()

    hand_ckpt_path = './ckpts/hand_flat_same/epoch=7.ckpt'
    object_ckpt_path = './ckpts/cup_s8/epoch=7.ckpt'

    load_ckpt(hand_nerf_coarse, hand_ckpt_path, model_name='nerf_coarse')
    load_ckpt(hand_nerf_fine, hand_ckpt_path, model_name='nerf_fine')
    load_ckpt(object_nerf_coarse, object_ckpt_path, model_name='nerf_coarse')
    load_ckpt(object_nerf_fine, object_ckpt_path, model_name='nerf_fine')

    hand_nerf_coarse.cuda().eval()
    hand_nerf_fine.cuda().eval()
    object_nerf_coarse.cuda().eval()
    object_nerf_fine.cuda().eval()
    hand_models = [hand_nerf_coarse, hand_nerf_fine]
    hand_embeddings = [hand_embedding_xyz, hand_embedding_dir]
    object_models = [object_nerf_coarse, object_nerf_fine]
    object_embeddings = [object_embedding_xyz, object_embedding_dir]

    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True)

    model = NeRF(hand_models, hand_embeddings, object_models, object_embeddings, mano_layer, ncomps, N_samples, N_importance).to(device)

    if opt.test:
        guidance = None # no need to load guidancemodel at test

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)
    else:
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        if opt.optim == 'adan':
            from utils.optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        model_key = './models/stable-diffusion-2-1-base'
        guidance = StableDiffusion(model_key, device=device, fp16=opt.fp16, vram_O=opt.vram_O, t_range=opt.t_range)

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, \
                          ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, \
                            eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test at the end
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)
