

# from src.models.image_model import DCVCRTImage
from src.models.lora_image_model import DCVCRTImage
from src.utils.transforms import ycbcr2rgb, rgb2ycbcr
from src.utils.utils import get_state_dict, replicated_pad, get_padding_size, AverageMeter 
from torchvision.transforms import ToTensor, ToPILImage
from src.metrics.metric import calculate_metrics
import argparse 
import os 
from pathlib import Path 
from PIL import Image 
import torch 
import lpips



EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_args():
    args = argparse.ArgumentParser(description="Test YCbCr Color Space Image Compression Model.")
    args.add_argument("--data_path", type=str, required=True)
    args.add_argument("--quant_level", nargs="+", type=int, default=[0, 8, 16, 32, 42, 63])
    args.add_argument("--cuda", action="store_true")
    args.add_argument("--save", action="store_true")
    args.add_argument("--save_dir", type=str, default="./results")
    args.add_argument("--model_path", type=str, default=None)
    args.add_argument("--lora_path", type=str, default=None)
    return args.parse_args() 



def padding_image(image, pad_size: int = 64):
    H, W = image.size()[-2:]
    pad_b, pad_r = get_padding_size(H, W, pad_size)
    pad_image = replicated_pad(image, pad_b, pad_r)
    return pad_image


def scan_images(folder_path):
    """
    扫描文件夹及其子目录，返回特定后缀的图像路径列表
    :param folder_path: str, 要扫描的根目录路径
    :param extensions: list, 允许的图像后缀名列表，如 ['.jpg', '.png']
    :return: list, 包含所有匹配图像绝对路径的列表
    """

    extensions = EXTENSIONS
 
    # 将后缀名统一转为小写，并确保前面带有 '.' 以便后续匹配
    valid_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}
    root_dir = Path(folder_path)
    image_paths = []
    # rglob('*') 递归遍历所有文件和目录
    for file_path in root_dir.rglob('*'):
        # 判断是否是文件，且后缀名在允许的列表中（忽略大小写）
        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
            # 返回绝对路径字符串
            image_paths.append(str(file_path.resolve()))
            
    return image_paths


def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tr = ToTensor()(image).unsqueeze(0)
    return img_tr 



@torch.no_grad()
def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    perceptual_loss = lpips.LPIPS(net="alex").to(device)    
    
    # log results
    results = []
    
    model = DCVCRTImage().to(device)
    model.load_state_dict(get_state_dict(args.model_path), strict=False)
    lora_ckpt = torch.load(args.lora_path, map_location="cpu")
    model.load_state_dict(lora_ckpt, strict=False)
    model.eval()
    
    image_paths = scan_images(args.data_path)
    
    if args.save:
        assert args.save_dir is not None, "save_dir must be provided if save is True"
        os.makedirs(args.save_dir, exist_ok=True)
    
    for qp in args.quant_level:
        
        cur_bpp, cur_psnr, cur_ssim = AverageMeter(), AverageMeter(), AverageMeter()
        cur_lpips = AverageMeter()
        
        qp_tr = torch.tensor(qp, dtype=torch.long).to(device)
        
        for image_path in image_paths:
            
            img_tr = read_image(image_path).to(device)
            B, C, H, W = img_tr.size()
            
            
            img_pad = padding_image(img_tr)
            img_ycbcr = rgb2ycbcr(img_pad)
            
            output = model(img_ycbcr, qp=qp_tr)
            
            img_recon = output["x_hat"]
            img_recon = ycbcr2rgb(img_recon)[:, :, :H, :W]
            
            psnr_val, ssim_val = calculate_metrics(img_tr, img_recon)
            
            bpp = output["bpp"].mean().item()
            
            lpips_val = perceptual_loss.forward(img_tr, img_recon, normalize=True).item()
            
            
            cur_bpp.update(bpp)
            cur_psnr.update(psnr_val)
            cur_ssim.update(ssim_val)
            cur_lpips.update(lpips_val)

            if args.save:
                img = ToPILImage()(img_recon[0])
                img.save(os.path.join(args.save_dir, os.path.basename(image_path).split(".")[0] + f"_qp{qp}.png"))
            
        results.append({
            "qp": qp,
            "bpp": cur_bpp.avg,
            "psnr": cur_psnr.avg,
            "ssim": cur_ssim.avg,
            "lpips": cur_lpips.avg,
        })
        print(f"QP: {qp}, Bpp: {cur_bpp.avg:.4f}, PSNR: {cur_psnr.avg:.4f}, SSIM: {cur_ssim.avg:.4f}, LPIPS: {cur_lpips.avg:.4f}")
    
    # if args.save:
    with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
        f.write("QP\tBpp\tPSNR\tSSIM\tLPIPS\n")
        for result in results:
            f.write(f"{result['qp']}\t{result['bpp']:.4f}\t{result['psnr']:.4f}\t{result['ssim']:.4f}\t{result['lpips']:.4f}\n")
    
    print("Test done.")
    


if __name__ == "__main__":
    args = parse_args()
    test(args)
    
    
        