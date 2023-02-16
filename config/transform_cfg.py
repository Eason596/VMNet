from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
)


transform_cfg = {
    'parse2022': {
        'train': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes='LPS'),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=500,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandRotate90d(keys=['image', 'label'], prob = 0.2),
            SpatialPadd(keys=['image', 'label'], spatial_size=(128, 128, 128), method='symmetric', mode='constant', constant_values=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                # spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]),
        'val': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes='LPS'),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2),mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ]),
        'test': Compose([
            LoadImaged(keys=["image"], ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes='LPS'),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2),mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        ])
    },

    '3Dircadb': {
        'train': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes='LPS'),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandRotate90d(keys=['image', 'label'], prob = 0.2),
            SpatialPadd(keys=['image', 'label'], spatial_size=(128, 128, 128), method='symmetric', mode='constant', constant_values=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                # spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]),
        'val': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes='LPS'),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=400, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ]),
        'test': Compose([
            LoadImaged(keys=["image"], ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes='LPS'),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        ])
    } ,



    'most': {
        'train': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandRotate90d(keys=['image', 'label'], prob = 0.2),
            SpatialPadd(keys=['image', 'label'], spatial_size=(128, 128, 128), method='symmetric', mode='constant', constant_values=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                # spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]),
        'val': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ]),
        'test': Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ])
    } 
}