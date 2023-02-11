from monai.transforms import Compose, LoadImaged, ToTensord, RandAffined, EnsureTyped
from monai.data import Dataset, DataLoader
import os
import torch


def get_dataloader(root, transform, batch_size=1, num_workers=8, pin_memory=True, drop_last=True):
    img_list = [os.path.join(root, 'img', i) for i in os.listdir(os.path.join(root, 'img'))]
    mask_list = [i.replace('img', 'mask') for i in img_list]
    data = [{'img': i, 'mask': m} for i, m in zip(img_list, mask_list)]
    dataset = Dataset(data=data, transform=transform)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last)
    return dataloader


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


