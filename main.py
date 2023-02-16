import yaml
import time
import os
import torch
from model.unet3d import UNet3D
from utils import get_cashedata_and_threaddataloader, get_dataset_files, train_loop, val_loop, test_loop, get_logger, mkcfgdir_and_if_resume_training
from config.transform_cfg import transform_cfg
from torch.nn import BCELoss
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss


MODEL_DICT = {
    'unet3d': UNet3D,
}

def main(cfg_path, mode='train'):
    # 读取配置文件
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
    log_save_path = cfg['log_save_path'] + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    # 获取数据
    train_files, val_files, test_files = get_dataset_files(cfg['dataset_name'])
    transfrom = transform_cfg[cfg['dataset_name']]
    train_loader = get_cashedata_and_threaddataloader(
        files=train_files, 
        transfrom=transfrom['train'], 
        cache_num=24, 
        cache_rate=1, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=cfg['num_workers'])
    val_loader = get_cashedata_and_threaddataloader(
        files=val_files, 
        transfrom=transfrom['val'],
        cache_num=8,
        cache_rate=1,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'])
    # test_loader = get_cashedata_and_threaddataloader(
    #     files=test_files, 
    #     transfrom=transfrom['test'],
    #     cache_num=8,
    #     cache_rate=1,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=cfg['num_workers'])

    

    # 模型和训练设置
    device = torch.device(cfg['device'])
    model = MODEL_DICT[cfg['model_name']](cfg).to(device)

    if cfg['load_checkpoint_from'] is not None and os.path.exists(cfg['load_checkpoint_from']):
        model.load_state_dict(torch.load(cfg['load_checkpoint_from']))

    if cfg['loss_mode'] == 1:
        loss_func = BCELoss()
    elif cfg['loss_mode'] == 2:
        loss_func = DiceLoss()
    elif cfg['loss_mode'] == 3:
        loss_func = DiceCELoss()
    elif cfg['loss_mode'] == 4:
        loss_func = DiceFocalLoss()

    if cfg['optimizer'] == 1:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])

    if cfg['lr_scheduler'] == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'], eta_min=1e-6)
    elif cfg['lr_scheduler'] == 2:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=1e-5)

    last_epoch = mkcfgdir_and_if_resume_training(mode, cfg_path, log_save_path, cfg)
    if mode == 'train':
        logger = get_logger(os.path.join(log_save_path, 'training.log'))
        for epoch in range(last_epoch, last_epoch + cfg['epoch']):
            train_loop(epoch, model, train_loader, optimizer, loss_func, scheduler, logger, device,
                       cfg['iters_to_accumulate'])
            if (epoch + 1) % cfg['val_step'] == 0:
                val_loop(epoch, model, val_loader, loss_func, logger, device, cfg['slide_inference'], eval(cfg['slide_crop_size']))
                torch.save(model.state_dict(), os.path.join(log_save_path, 'checkpoint', 'epoch_{}.pth'.format(epoch)))
    elif mode == 'test':
        save_img_path = os.path.join(cfg['load_from'].split('checkpoint')[0], 'imgs')
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        test_loop(model, val_loader, device, save_img_path, cfg['slide_inference'], eval(cfg['slide_crop_size']))


if __name__ == '__main__':
    mode = 'train'
    # mode = 'test'
    cfg_path = 'config/unet3d_most.yaml'

    main(cfg_path, mode)
