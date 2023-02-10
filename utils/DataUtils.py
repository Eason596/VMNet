import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, Spacingd, CropForegroundd, AddChanneld, RandCropByPosNegLabeld, EnsureTyped
from albumentations import clahe


class Ircadb3D:
    def __init__(self,
                 root='/home/yjiang/code/data/3Dircadb/3Dircadb1',
                 merge_path='/home/yjiang/code/data/3Dircadb/merge'):
        '''
        目录结构
        3Dircadb
        |-- 3Dircadb1
        :param root: 3Dircadb1目录所在位置
        :param merge_path:
        '''
        self.root = root
        self.merge_path = merge_path

    def merge(self):
        '''
        将原始3DIrcadb1数据集20例病人的原始CT图像和血管标签进行合并，将结果保存到save_path,生成文件如下
        merge_path
        |-- img
        |--  |-- 1.nii.gz
        |--  |-- ...
        |-- mask
        |--  |-- 1.nii.gz
        |--  |-- ...
        '''
        mask_dirs = ['venoussystem', 'artery', 'venacava', 'portalvein', 'portalvein1']

        if not os.path.exists(os.path.join(self.merge_path, 'img')):
            os.makedirs(os.path.join(self.merge_path, 'img'))
            os.makedirs(os.path.join(self.merge_path, 'mask'))
        print('start merge')
        for i in tqdm(os.listdir(self.root)):
            lenth = len(os.listdir(os.path.join(self.root, i, 'PATIENT_DICOM')))
            img_list = []
            mask_list = []
            sample = sitk.ReadImage(os.path.join(self.root, i, 'PATIENT_DICOM', 'image_0'))
            origin = sample.GetOrigin()
            spacing = sample.GetSpacing()
            direction = sample.GetDirection()
            for idx in range(lenth):
                img_list.append(sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join(self.root, i, 'PATIENT_DICOM', 'image_{}'.format(idx)))))
                mask = np.zeros((1, 512, 512))
                for mask_dir in mask_dirs:
                    mask_path = os.path.join(self.root, i, 'MASKS_DICOM', mask_dir, 'image_{}'.format(idx))
                    if os.path.exists(mask_path):
                        mask += sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                mask_list.append(mask)
            img = sitk.GetImageFromArray(np.concatenate(img_list))
            img.SetSpacing(spacing)
            img.SetDirection(direction)
            img.SetOrigin(origin)

            mask = sitk.GetImageFromArray(np.concatenate(mask_list))
            mask.SetSpacing(spacing)
            mask.SetDirection(direction)
            mask.SetOrigin(origin)

            sitk.WriteImage(img, os.path.join(self.merge_path, 'img', '{}.nii.gz'.format(i.split('.')[1])))
            sitk.WriteImage(mask, os.path.join(self.merge_path, 'mask', '{}.nii.gz'.format(i.split('.')[1])))

    def data_gen(self,
                 save_path='/home/yjiang/code/data/3Dircadb/process',
                 spatial_size=(128, 128, 128),
                 respacing=(0.5, 0.5, 0.5),
                 clip=(0, 400),
                 num_samples=4,
                 patient_list=range(1, 21)):
        '''
        将合并后的数据进行重采样和剪切，生成训练用数据，生成结果如下
        save_path
        |-- img
        |--  |-- patinet_1_slice_0.nii.gz
        |--  |-- ...
        |--  |-- patinet_1_slice_{num_samples}.nii.gz
        |--  |-- ...
        |-- mask
        |--  |-- patinet_1_slice_0.nii.gz
        |--  |-- ...
        |--  |-- patinet_1_slice_{num_samples}.nii.gz
        |--  |-- ...
        :param save_path: 文件保存零
        :param spatial_size: 裁剪图像大小
        :param respacing: 重采样大小
        :param clip: HU截断值
        :param num_samples: 每个病人裁剪数据块个数
        :param patient_list: 病人列表
        :return:
        '''
        min_hu, max_hu = clip
        if not os.path.exists(os.path.join(save_path, 'img')):
            os.makedirs(os.path.join(save_path, 'img'))
            os.makedirs(os.path.join(save_path, 'mask'))

        print('start generate')
        for i in tqdm(patient_list):
            img_path = os.path.join(self.merge_path, 'img', '{}.nii.gz'.format(i))
            mask_path = os.path.join(self.merge_path, 'mask', '{}.nii.gz'.format(i))

            transform = Compose([
                LoadImaged(keys=['img', 'mask'], ensure_channel_first=True),
                Spacingd(keys=['img', 'mask'], pixdim=respacing, mode=("bilinear", "nearest")),
                RandCropByPosNegLabeld(keys=["img", "mask"], label_key='mask', spatial_size=spatial_size, neg=0,
                                       num_samples=num_samples, allow_smaller=False),
                EnsureTyped(keys=["img", "mask"], data_type='numpy', dtype=np.int64)
            ])
            process = transform({'img': img_path, 'mask': mask_path})
            print(process[0]['img'].shape)
            for j in range(num_samples):
                img = (((np.clip(process[j]['img'][0], min_hu, max_hu) - min_hu) / (max_hu - min_hu)) * 255).astype(
                    np.uint8)
                img = np.stack([clahe(img[:, :, k], clip_limit=4) for k in range(img.shape[2])], axis=2)
                mask = process[j]['mask'][0] // 255
                sitk.WriteImage(sitk.GetImageFromArray(img),
                                os.path.join(save_path, 'img', 'patient_{}_slice_{}.nii.gz'.format(i, j)))
                sitk.WriteImage(sitk.GetImageFromArray(mask),
                                os.path.join(save_path, 'mask', 'patient_{}_slice_{}.nii.gz'.format(i, j)))


if __name__ == '__main__':
    # 生成3Didcadb训练数据 前15号病人用于训练 后5个用于测试
    idcadb_dataset = Ircadb3D(root='/home/yjiang/code/data/3Dircadb/3Dircadb1', merge_path='/home/yjiang/code/data/3Dircadb/merge')
    idcadb_dataset.merge()
    idcadb_dataset.data_gen(save_path='/home/yjiang/code/data/3Dircadb/process_128',
                            spatial_size=(128, 128, 128),
                            respacing=(0.5, 0.5, 0.5),
                            clip=(0, 400),
                            num_samples=4,
                            patient_list=range(1, 16))
    idcadb_dataset.data_gen(save_path='/home/yjiang/code/data/3Dircadb/process_512',
                            spatial_size=(512, 512, 8),
                            respacing=(0.5, 0.5, 0.5),
                            clip=(0, 400),
                            num_samples=4,
                            patient_list=range(1, 16))
