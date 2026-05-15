"""VAE reconstruction demo script."""
import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage

from torchvision.transforms import InterpolationMode, transforms
import torch.nn.functional as F

from models import VQVAE, build_vae


def pil_loader(path):
    """Load image from path and convert to RGB."""
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def normalize_01_into_pm1(x):
    """Normalize x from [0, 1] to [-1, 1]."""
    return x.add(x).add_(-1)


def get_transform(mid_reso=1.0, final_reso=256):
    """Build validation transform pipeline."""
    mid_reso = round(mid_reso * final_reso)
    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    return transforms.Compose(val_aug)


def get_reconstruct(img_path, vae, val_aug):
    """Reconstruct image using VAE."""
    img = pil_loader(img_path)
    img_tensor = val_aug(img)
    img_tensor = img_tensor.to(torch.bfloat16)

    vae = vae.to(torch.bfloat16)
    tmp = vae.img_to_reconstructed_img(img_tensor.unsqueeze(0).cuda())
    tmp = [item.add_(1).mul_(0.5) for item in tmp]
    img_final = tmp[-1]
    tmp = torch.cat(tmp, dim=0)

    chw = torchvision.utils.make_grid(tmp, nrow=10, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().float().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))

    img_final = PImage.fromarray(
        (img_final[0].permute(1, 2, 0).mul_(255).cpu().float().numpy()).astype(np.uint8)
    )
    return chw, img_final


if __name__ == '__main__':
    # Disable default parameter init for faster speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    # Build VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = build_vae(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device,
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print('VAE model ready.')

    # Set random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable TF32 for faster computation
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Run reconstruction
    val_aug = get_transform(mid_reso=1.0)

    # Use relative path to assets directory
    script_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.dirname(osp.dirname(osp.dirname(script_dir)))
    img_path = osp.join(project_root, 'assets', 'image1.jpg')

    chw, img_final = get_reconstruct(img_path, vae, val_aug)
    chw.save('reconstruct_grid.png')
    img_final.save('reconstruct_final.png')
    print(f'Saved results to reconstruct_grid.png and reconstruct_final.png')