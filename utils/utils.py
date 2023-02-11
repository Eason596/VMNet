from monai.transforms import Compose, LoadImaged, ToTensord, RandAffined, EnsureTyped
from monai.data import Dataset, DataLoader
import os
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt


def get_dataloader(root, transform, batch_size=1, num_workers=8, pin_memory=True, drop_last=True):
    img_list = [os.path.join(root, 'img', i) for i in os.listdir(os.path.join(root, 'img'))]
    mask_list = [i.replace('img', 'mask') for i in img_list]
    data = [{'img': i, 'mask': m} for i, m in zip(img_list, mask_list)]
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last)
    return dataloader


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
    history.columns = ['epoch', 'loss', 'acc', 'dice', 'hd']
    history['mode'] = history['epoch'].str.slice(1, 2)
    history['epoch'] = history['epoch'].str.extract('(\d+)')
    history['loss'] = history['loss'].str.extract('(\d+.\d+)').astype('float')
    history['acc'] = history['acc'].str.extract('(\d+.\d+)').astype('float')
    history['dice'] = history['dice'].str.extract('(\d+.\d+)').astype('float')
    history['hd'] = history['hd'].str.extract('(\d+.\d+)').astype('float')
    step_train = history[history['mode'] == 't']
    step_val = history[history['mode'] == 'v']
    epoch_train = step_train.groupby('epoch').mean()
    epoch_train.index = epoch_train.index.astype('int')
    epoch_train.sort_index(inplace=True)
    epoch_val = step_val.groupby('epoch').mean()
    epoch_val.index = epoch_val.index.astype('int')
    epoch_val.sort_index(inplace=True)
    size = len(epoch_train.columns)
    plt.figure(figsize=(15, 25))
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


if __name__ == '__main__':
    root = '/home/yjiang/code/data/3Dircadb/process_128'
    torch.multiprocessing.set_sharing_strategy('file_system')
    transform = Compose([
        LoadImaged(keys=['img', 'mask'], ensure_channel_first=True),
        RandAffined(
            keys=["img", "mask"],
            mode=("bilinear", "nearest"),
            prob=1.0,
            spatial_size=(128, 128, 128),
            translate_range=(2, 2, 2),
            rotate_range=(15, 15, 15),
            scale_range=(0.2, 0.2, 0.2),
            padding_mode="zeros",
        ),
        ToTensord(keys=['img', 'mask'])
    ])
    dataloader = get_dataloader(root, transform, batch_size=4, num_workers=8)

    for data in dataloader:
        i = data['img'] / 255
        m = data['mask']
        print(i.shape)
        print(m.shape)
        print(i.dtype)
        print(m.dtype)
        print(torch.unique(i))
        print(torch.unique(m))
        break


