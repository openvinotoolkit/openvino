from pathlib import Path
from ..representation import SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..config import PathField, StringField, BoolField
from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


train_meta = {
    'label_map': {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
        7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    },
    'segmentation_colors': (
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
        (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
        (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ),
}

full_dataset_meta = {
    'segmentation_colors' : (
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128),
        (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
        (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
        (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ),
    'label_map': {
        0: 'unlabeled', 1:  'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static', 5: 'dynamic',
        6: 'ground', 7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track', 11: 'building', 12: 'wall',
        13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'polegroup', 19: 'traffic light',
        20: 'traffic sign', 21: 'vegetation', 22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car',
        27: 'truck', 28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle',
        -1: 'license plate'
    }
}


class CityscapesConverterConfig(BaseFormatConverterConfig):
    dataset_root_dir = PathField(is_directory=True)
    images_subfolder = StringField(optional=True)
    masks_subfolder = StringField(optional=True)
    masks_suffix = StringField(optional=True)
    images_suffix = StringField(optional=True)
    use_full_label_map = BoolField(optional=True)


class CityscapesConverter(BaseFormatConverter):
    __provider__ = 'cityscapes'

    _config_validator_type = CityscapesConverterConfig

    def configure(self):
        self.dataset_root = self.config['dataset_root_dir']
        self.images_dir = self.config.get('images_subfolder', 'imgsFine/leftImg8bit/val')
        self.masks_dir = self.config.get('masks_subfolder', 'gtFine/val')
        self.masks_suffix = self.config.get('masks_suffix', '_gtFine_labelTrainIds')
        self.images_suffix = self.config.get('images_suffix', '_leftImg8bit')
        self.use_full_label_map = self.config.get('use_full_label_map', False)


    def convert(self):
        images = list(self.dataset_root.rglob(r'{}/*/*{}.png'.format(self.images_dir, self.images_suffix)))
        annotations = []
        for image in images:
            identifier = str(Path(self.images_dir).joinpath(*image.parts[-2:]))
            mask = Path(self.masks_dir) / image.parts[-2] / self.masks_suffix.join(
                str(image.name).split(self.images_suffix)
            )
            annotations.append(SegmentationAnnotation(identifier, mask, mask_loader=GTMaskLoader.PILLOW))

        return annotations, full_dataset_meta if self.use_full_label_map else train_meta
