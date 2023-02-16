import os
import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
from PIL import Image
import logging
import shutil
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric, ConfusionMatrixMetric, HausdorffDistanceMetric
from monai.data import CacheDataset, ThreadDataLoader
import yaml
import pandas as pd

def mkcfgdir_and_if_resume_training(mode, cfg_path, log_save_path, cfg):
    last_epoch = 0
    if mode == 'train' and not os.path.exists(log_save_path):
        os.makedirs(os.path.join(log_save_path, 'checkpoint'))
        shutil.copy(cfg_path, os.path.join(log_save_path, cfg_path.split('/')[-1]))
        shutil.copy('config/transform_cfg.py', os.path.join(log_save_path, 'transform_cfg.py'))

    if mode == 'train' and cfg['resume']:
        if cfg['load_from'] is None:
            print('resume error')
        last_path = cfg['load_from'].split('checkpoint')[0]
        shutil.copy(os.path.join(last_path, 'training.log'), os.path.join(log_save_path, 'training.log'))
        with open(os.path.join(last_path, cfg_path.split('/')[-1]), 'r', encoding='utf-8') as f:
            last_cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
            last_epoch = last_cfg['epoch']
        with open(os.path.join(last_path, cfg_path.split('/')[-1]), 'w', encoding='utf-8') as f:
            last_cfg['epoch'] = last_epoch + cfg['epoch']
            yaml.dump(last_cfg, f, default_flow_style=False)
    return last_epoch
            
def get_dataset_files(dataset_name):
    if dataset_name == 'parse2022':
        root = '/home/yjiang/code/data/parse2022/raw_data/train/'
        dirs = sorted([i for i in os.listdir(root)])
        train_dirs = dirs[:80]
        val_dirs = dirs[80:]
        imgs = [os.path.join(root, i, 'image', i+'.nii.gz') for i in train_dirs]
        masks = [os.path.join(root, i, 'label', i+'.nii.gz') for i in train_dirs]
        train_files = [{"image": img, "label": mask} for img, mask in zip(imgs, masks)]
        imgs = [os.path.join(root, i, 'image', i+'.nii.gz') for i in val_dirs]
        masks = [os.path.join(root, i, 'label', i+'.nii.gz') for i in val_dirs]
        val_files = [{"image": img, "label": mask} for img, mask in zip(imgs, masks)]
        test_files = [{"image": img} for img in imgs]
    elif dataset_name == '3Dircadb':
        root = '/home/yjiang/code/data/3Dircadb/merge'
        imgs = [os.path.join(root, 'img', i) for i in os.listdir(os.path.join(root, 'img'))]
        masks = [os.path.join(root, 'mask', i) for i in os.listdir(os.path.join(root, 'mask'))]
        train_files = [{"image": img, "label": mask} for img, mask in zip(imgs[:15], masks[:15])]
        val_files = [{"image": img, "label": mask} for img, mask in zip(imgs[15:], masks[15:])]
        test_files = [{"image": img} for img in imgs[15:]]
    elif dataset_name == 'most':
        root = '/home/yjiang/code/data/most'
        imgs = [os.path.join(root, 'most_crop', 'img', i) for i in os.listdir(os.path.join(root, 'most_crop', 'img'))]
        masks = [os.path.join(root, 'most_crop', 'mask', i) for i in os.listdir(os.path.join(root, 'most_crop','mask'))]
        train_files = [{"image": img, "label": mask} for img, mask in zip(imgs, masks)]
        files = sorted(os.listdir(root+'/raw'))[20:40]
        imgs = [os.path.join(root, 'raw', i) for i in files if 'img' in i]
        masks = [os.path.join(root, 'raw', i) for i in files if 'mask' in i]
        val_files = [{"image": img, "label": mask} for img, mask in zip(imgs, masks)]
        files = sorted(os.listdir(root+'/raw'))[40:]
        imgs = [os.path.join(root, 'raw', i) for i in files if 'img' in i]
        masks = [os.path.join(root, 'raw', i) for i in files if 'mask' in i]
        test_files = [{"image": img, "label": mask} for img, mask in zip(imgs, masks)]
    else:
        raise NotImplementedError
    
    return train_files, val_files, test_files

def get_cashedata_and_threaddataloader(files, transfrom, cache_num=24, cache_rate=1, batch_size=1, shuffle=False, num_workers=8):
    dataset = CacheDataset(data=files, transform=transfrom, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
    dataloader = ThreadDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

def get_metirc(y_pred, y_true):
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
    cfm_metric = ConfusionMatrixMetric(metric_name=['sensitivity', 'specificity', 'accuracy'], include_background=True,
                                       reduction="mean")
    hd_metric = HausdorffDistanceMetric(percentile=95, reduction="mean")
    dice_metric(y_pred=y_pred, y=y_true)
    cfm_metric(y_pred=y_pred, y=y_true)
    hd_metric(y_pred=y_pred, y=y_true)
    dice = dice_metric.aggregate().item()
    sensitivity, specificity, accuracy = cfm_metric.aggregate()
    sensitivity, specificity, accuracy = sensitivity.item(), specificity.item(), accuracy.item()
    hd95 = hd_metric.aggregate().item()
    dice_metric.reset()
    cfm_metric.reset()
    hd_metric.reset()
    return sensitivity, specificity, accuracy, dice, hd95

def train_loop(epoch, model, train_loader, optimizer, loss_func, scheduler, logger, device, iters_to_accumulate=1):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description("[training {}]".format(epoch))

    for i, data in enumerate(pbar):
        img = data['image'].to(device)
        mask = data['label'].to(device)

        predict = torch.sigmoid(model(img))
        loss = loss_func(predict, mask) / iters_to_accumulate
        loss.backward()
        if (i + 1) % iters_to_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
        predict = torch.where(predict < 0.5, 0, 1)

        sensitivity, specificity, accuracy, dice, hd95 = get_metirc(predict.long(), mask.long())

        loss_value = loss.item() * iters_to_accumulate

        # pbar.set_postfix(loss=loss_value, acc=acc, dice=dice, hd=hd)
        logger.info(
            '[train_epoch:{}] loss:{:.4f} sensitivity:{:.4f} specificity:{:.4f} accuracy:{:.4f} dice:{:.4f} hd95:{:.4f}'.format(epoch, loss_value, sensitivity, specificity, accuracy, dice, hd95))

    scheduler.step()

def val_loop(epoch, model, val_loader, loss_func, logger, device, slide=False, crop=(160, 160, 160)):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader)
        pbar.set_description("[start val]")
        for data in pbar:
            img = data['image']
            mask = data['label']
            if slide:
                # predict = slide_inference(model, img, device, stride=stride, crop=crop)
                predict = sliding_window_inference(img, roi_size=crop, sw_batch_size=1, overlap=0.25, predictor=model, sw_device=device, device=device, progress=False)
                predict = torch.sigmoid(predict)
            else:
                predict = torch.sigmoid(model(img.to(device)))
            predict = predict.cpu()
            loss = loss_func(predict, mask)
            predict = torch.where(predict < 0.5, 0, 1)
            sensitivity, specificity, accuracy, dice, hd95 = get_metirc(predict.long(), mask.long())
            logger.info(
                '[val_epoch:{}] loss:{:.4f} sensitivity:{:.4f} specificity:{:.4f} accuracy:{:.4f} dice:{:.4f} hd95:{:.4f}'.format(
                    epoch, loss, sensitivity, specificity, accuracy, dice, hd95))

def test_loop(model, dataloader, device, save_path, slide=False, crop=(128, 128, 128)):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description("[start test]")
        for step, data in enumerate(pbar):
            img = data['image']
            mask = data['label']
            if slide:
                predict = sliding_window_inference(img, roi_size=crop, sw_batch_size=1, overlap=0.25, predictor=model, sw_device=device, device=device, progress=False)
                predict = torch.sigmoid(predict)
            else:
                predict = torch.sigmoid(model(img.to(device)))
            predict = predict.cpu()
            predict[predict >= 0.5] = 255
            predict[predict < 0.5] = 0
            img = img.detach().cpu().numpy() * 255
            mask = mask.detach().cpu().numpy() * 255
            predict = predict.detach().cpu().numpy()
            if slide:
                sitk.WriteImage(sitk.GetImageFromArray(img.astype(np.uint8)), os.path.join(save_path, '{}_img.nii.gz'.format(step)))
                sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), os.path.join(save_path, '{}_mask.nii.gz'.format(step)))
                sitk.WriteImage(sitk.GetImageFromArray(predict.astype(np.uint8)), os.path.join(save_path, '{}_predict.nii.gz'.format(step)))
            else:
                for batch, (m, i, p) in enumerate(zip(mask, img, predict)):
                    for index, slice in enumerate(zip(m[0], i[0], p[0])):
                        Image.fromarray(np.concatenate(slice, axis=1).astype(np.uint8)).convert('L').save(
                            os.path.join(save_path, '{}_{}_{}.png'.format(step, batch, index)))

def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log = logging.FileHandler(filename, 'a', encoding='utf-8')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
    log.setFormatter(formatter)
    logger.addHandler(log)
    return logger


def plot_log(filename):
    history = pd.read_csv(filename, sep=' ', header=None).iloc[:, 5:-1]
    history.columns = ['epoch', 'loss', 'sensitivity', 'specificity', 'accuracy', 'dice', 'hd95']
    history['mode'] = history['epoch'].str.slice(1, 2)
    history['epoch'] = history['epoch'].str.extract('(\d+)')
    history['loss'] = history['loss'].str.extract('(\d+.\d+)').astype('float')
    history['sensitivity'] = history['sensitivity'].str.extract('(\d+.\d+)').astype('float')
    history['specificity'] = history['specificity'].str.extract('(\d+.\d+)').astype('float')
    history['accuracy'] = history['accuracy'].str.extract('(\d+.\d+)').astype('float')
    history['dice'] = history['dice'].str.extract('(\d+.\d+)').astype('float')
    history['hd95'] = history['hd95'].str.extract(':(\d+.\d+)').astype('float')
    step_train = history[history['mode'] == 't']
    step_val = history[history['mode'] == 'v']
    epoch_train = step_train.groupby('epoch').mean()
    epoch_train.index = epoch_train.index.astype('int')
    epoch_train.sort_index(inplace=True)
    epoch_val = step_val.groupby('epoch').mean()
    epoch_val.index = epoch_val.index.astype('int')
    epoch_val.sort_index(inplace=True)
    size = len(epoch_train.columns)
    plt.figure(figsize=(15, 35))
    for idx, i in enumerate(epoch_train.columns):
        plt.subplot(size, 2, idx * 2 + 1)
        epoch_train[i].plot()
        plt.title('train {}'.format(i))
        plt.subplot(size, 2, idx * 2 + 2)
        epoch_val[i].plot(label='val {}'.format(i))
        plt.title('val {}'.format(i))
    plt.savefig(os.path.join('/'.join(filename.split('/')[:-1]), filename.split('/')[-1].split('.')[0] + '.png'))

    print('train dice max  epcoh', epoch_train['dice'].argmax(), '\n', epoch_train.iloc[epoch_train['dice'].argmax()])
    print('train loss min  epcoh', epoch_train['loss'].argmin(), '\n', epoch_train.iloc[epoch_train['loss'].argmin()])
    print('val dice max  epcoh', epoch_val['dice'].argmax(), '\n', epoch_val.iloc[epoch_val['dice'].argmax()])
    print('val loss min  epcoh', epoch_val['loss'].argmin(), '\n', epoch_val.iloc[epoch_val['loss'].argmin()])


def data_gen_3Dircadb(root, save_path):
    mask_dirs = ['venoussystem', 'artery', 'venacava', 'portalvein', 'portalvein1']
    if not os.path.exists(os.path.join(save_path, 'img')):
        os.makedirs(os.path.join(save_path, 'img'))
        os.makedirs(os.path.join(save_path, 'mask'))
    print('start merge')
    for i in tqdm(os.listdir(root)):
        lenth = len(os.listdir(os.path.join(root, i, 'PATIENT_DICOM')))
        img_list = []
        mask_list = []
        sample = sitk.ReadImage(os.path.join(root, i, 'PATIENT_DICOM', 'image_0'))
        origin = sample.GetOrigin()
        spacing = sample.GetSpacing()
        direction = sample.GetDirection()
        for idx in range(lenth):
            img_list.append(sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(root, i, 'PATIENT_DICOM', 'image_{}'.format(idx)))))
            mask = np.zeros((1, 512, 512))
            for mask_dir in mask_dirs:
                mask_path = os.path.join(root, i, 'MASKS_DICOM', mask_dir, 'image_{}'.format(idx))
                if os.path.exists(mask_path):
                    mask += sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask[mask > 0] = 1
            mask_list.append(mask)
        img = sitk.GetImageFromArray(np.concatenate(img_list))
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        img.SetOrigin(origin)

        mask = sitk.GetImageFromArray(np.concatenate(mask_list))
        mask.SetSpacing(spacing)
        mask.SetDirection(direction)
        mask.SetOrigin(origin)

        sitk.WriteImage(img, os.path.join(save_path, 'img', '{}.nii.gz'.format(i.split('.')[1])))
        sitk.WriteImage(mask, os.path.join(save_path, 'mask', '{}.nii.gz'.format(i.split('.')[1])))

def data_gen_most(root, save_path):
    if not os.path.exists(os.path.join(save_path, 'img')):
        os.makedirs(os.path.join(save_path, 'img'))
        os.makedirs(os.path.join(save_path, 'mask'))
    
    crop = (143, 571, 571)
    stride = (143, 571, 571)
    files = sorted(os.listdir(root))[:20]
    for file in tqdm(files):
        if 'img' in file:
            img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, file)))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, file.replace('img', 'mask'))))
            mask[mask == 255] = 1
            N, H, W = img.shape
            n_stride, h_stride, w_stride = stride
            n_crop, h_crop, w_crop  = crop

            h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1
            n_grids = max(N - n_crop + n_stride - 1, 0) // n_stride + 1
            
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    for n_idx in range(n_grids):
                        y1 = h_idx * h_stride
                        x1 = w_idx * w_stride
                        n1 = n_idx * n_stride

                        y2 = min(y1 + h_crop, H)
                        x2 = min(x1 + w_crop, W)
                        n2 = min(n1 + n_crop, N)

                        y1 = max(y2 - h_crop, 0)
                        x1 = max(x2 - w_crop, 0)
                        n1 = max(n2 - n_crop, 0)

                        crop_img = img[n1:n2, y1:y2, x1:x2]
                        crop_mask = mask[n1:n2, y1:y2, x1:x2]
                        sitk.WriteImage(sitk.GetImageFromArray(crop_img), os.path.join(save_path, 'img', '{}_{}_{}_{}.nii.gz'.format(file.split('.')[0], n_idx, h_idx, w_idx)))
                        sitk.WriteImage(sitk.GetImageFromArray(crop_mask), os.path.join(save_path, 'mask', '{}_{}_{}_{}.nii.gz'.format(file.split('.')[0], n_idx, h_idx, w_idx)))

if __name__ == '__main__':
    print('start')

    train_files, val_files, test_files = get_dataset_files('most')
    print(train_files)
    print(val_files)
    print(test_files)


    # print('generate most dataset')
    # data_gen_most('/home/yjiang/code/data/most/raw', '/home/yjiang/code/data/most/most_crop')

    # print('plot log test')
    # plot_log('train_result/unet3d_3Dircadb2023-02-15_01-37-46/training.log')