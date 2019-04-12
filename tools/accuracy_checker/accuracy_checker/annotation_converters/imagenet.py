import numpy as np

from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_txt, get_path

from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class ImageNetFormatConverterConfig(BaseFormatConverterConfig):
    annotation_file = PathField()
    labels_file = PathField(optional=True)
    has_background = BoolField(optional=True)


class ImageNetFormatConverter(BaseFormatConverter):
    __provider__ = 'imagenet'

    _config_validator_type = ImageNetFormatConverterConfig

    def configure(self):
        self.annotation_file = self.config['annotation_file']
        self.labels_file = self.config.get('labels_file')
        self.has_background = self.config.get('has_background', False)

    def convert(self):
        annotation = []
        for image in read_txt(get_path(self.annotation_file)):
            image_name, label = image.split()
            label = np.int64(label) if not self.has_background else np.int64(label) + 1
            annotation.append(ClassificationAnnotation(image_name, label))
        meta = self._create_meta(self.labels_file, self.has_background) if self.labels_file else None

        return annotation, meta

    @staticmethod
    def _create_meta(labels_file, has_background=False):
        meta = {}
        labels = {}
        for i, line in enumerate(read_txt(get_path(labels_file))):
            index_for_label = i if not has_background else i + 1
            line = line.strip()
            label = line[line.find(' ') + 1:]
            labels[index_for_label] = label

        if has_background:
            labels[0] = 'background'
            meta['backgound_label'] = 0

        meta['label_map'] = labels

        return meta
