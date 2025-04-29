from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np

import datasets.pixabay_dataset

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
import time
from PIL import Image


def is_dic(x):
    return type(x) == type([])



def save_separate_outputs(inputs, preds, bg_dir, mask_dir, img_fn):
    """
    使用PIL.Image保存背景和掩码预测结果
    Args:
        bg_pred: torch.Size([1, 3, 256, 256])  # [batch, channels, height, width]
        mask_pred: torch.Size([1, 1, 256, 256])
    """
    os.makedirs(bg_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    bg_pred, mask_pred = preds['bg'], preds['mask'][0]  # 取第一个mask

    def tensor_to_numpy(tensor, is_mask=False):
        """将张量转换为HWC格式的numpy数组"""
        arr = tensor.squeeze(0).detach().cpu().numpy() 
        if is_mask:
            arr = (arr * 255).clip(0, 255) 
        else:
            arr = (arr * 255).clip(0, 255)  
        return arr.transpose(1, 2, 0).astype(np.uint8) 
    
    # 处理背景图像 (RGB)
    bg_np = tensor_to_numpy(bg_pred)
    if bg_np.shape[-1] == 1: 
        bg_np = np.repeat(bg_np, 3, axis=-1) 
    
    # 处理掩码图像 (单通道)
    mask_np = tensor_to_numpy(mask_pred, is_mask=True)
    if mask_np.shape[-1] == 3: 
        mask_np = mask_np.mean(axis=-1, keepdims=True)  
    
    filename = os.path.splitext(os.path.basename(img_fn))[0]
    
    bg_path = os.path.join(bg_dir, f"{filename}.png")
    Image.fromarray(bg_np, 'RGB').save(bg_path)
    print(f"已保存背景: {bg_path}")
    
    # 保存掩码
    mask_path = os.path.join(mask_dir, f"{filename}.png")
    Image.fromarray(mask_np.squeeze(), 'L').save(mask_path)  # 'L'模式表示单通道
    print(f"已保存掩码: {mask_path}")


def main(args):
    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    elif args.dataset == 'pixabay':
        dataset_func = datasets.PixabayDataset
    
    val_loader = torch.utils.data.DataLoader(dataset_func('test',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    data_loaders = (None,val_loader)

    Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

    
    model = Machine
    model.model.eval()
    print("==> testing VM model ")
    print(model.device)
    rmses = AverageMeter()
    rmsews = AverageMeter()
    ssimesx = AverageMeter()
    psnresx = AverageMeter()
    maskIoU = AverageMeter()
    maskF1 = AverageMeter()
    prime_maskIoU = AverageMeter()
    prime_maskF1 = AverageMeter()
    processTime = AverageMeter()

    prediction_dir = os.path.join(args.checkpoint,'rst')
    if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
    
    save_flag = True
    with torch.no_grad():
        for i, batches in enumerate(model.val_loader):

            inputs = batches['image'].to(model.device)
            target = batches['target'].to(model.device)
            mask =batches['mask'].to(model.device)
            img_path = batches['img_path']

            # select the outputs by the giving arch
            start_time = time.time()
            outputs = model.model(model.norm(inputs))
            process_time = time.time() - start_time
            processTime.update((process_time*1000), inputs.size(0))

            imoutput,immask_all,imwatermark = outputs
            imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            
            immask = immask_all[0]

            imfinal =imoutput*immask + model.norm(inputs)*(1-immask)
            psnrx = 10 * log10(1 / F.mse_loss(imfinal,target).item())       
            # ssimx = ssim(final_np, target_np, multichannel=True)
            ssimx = pytorch_ssim.ssim(imfinal, target)
            
            
            
            rmsex = compute_RMSE(imfinal, target, mask, is_w=False)
            rmsewx = compute_RMSE(imfinal, target, mask, is_w=True)
            rmses.update(rmsex, inputs.size(0))
            rmsews.update(rmsewx, inputs.size(0))
            psnresx.update(psnrx, inputs.size(0))
            ssimesx.update(ssimx, inputs.size(0))


            main_mask = immask_all[1::2]
            comp_mask = immask_all[2::2]
            out_mask = main_mask[-1]
            comp_mask = comp_mask[-1]
            
            comp_sets = []
            prime_mask_pred = torch.where(out_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
            mask_pred = torch.where(comp_mask > 0.5, torch.ones_like(out_mask), torch.zeros_like(out_mask)).to(out_mask.device)
           
            iou = compute_IoU(prime_mask_pred, mask)
            prime_maskIoU.update(iou)
            f1 = FScore(prime_mask_pred, mask).item()
            prime_maskF1.update(f1, inputs.size(0))

            iou = compute_IoU(mask_pred, mask)
            maskIoU.update(iou)
            f1 = FScore(mask_pred, mask).item()
            maskF1.update(f1, inputs.size(0))

            bg_dir = os.path.join(prediction_dir, 'SLBR_bg(pixabay)')
            mask_dir = os.path.join(prediction_dir, 'SLBR_mask(pixabay)')

            os.makedirs(bg_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            if save_flag:
                save_separate_outputs(
                    inputs={'I':inputs, 'bg':target,  'mask':mask}, 
                    preds={'bg':imfinal, 'mask':immask_all}, 
                    bg_dir=bg_dir, 
                    mask_dir=mask_dir, 
                    img_fn=img_path[0] 
                )
            if i % 100 == 0:
                print("Batch[%d/%d]| PSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(i,len(model.val_loader),psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("Total:\nPSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("DONE.\n")


if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())
    
