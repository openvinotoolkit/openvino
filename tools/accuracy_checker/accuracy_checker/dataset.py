"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
import numpy as np

from .annotation_converters import BaseFormatConverter, save_annotation, make_subset
from .data_readers import BaseReader, DataReaderField
from .config import ConfigValidator, StringField, PathField, ListField, DictField, BaseField, NumberField, ConfigError
from .utils import JSONDecoderWithAutoConversion, read_json, get_path, contains_all
from .representation import BaseRepresentation


class DataRepresentation:
    def __init__(self, data, meta=None, identifier=''):
        self.identifier = identifier
        self.data = data
        self.metadata = meta or {}
        if np.isscalar(data):
            self.metadata['image_size'] = 1
        elif isinstance(data, list) and np.isscalar(data[0]):
            self.metadata['image_size'] = len(data)
        else:
            self.metadata['image_size'] = data.shape if not isinstance(data, list) else data[0].shape


class DatasetConfig(ConfigValidator):
    """
    Specifies configuration structure for dataset
    """
    name = StringField()
    annotation = BaseField(optional=True)
    data_source = PathField()
    dataset_meta = BaseField(optional=True)
    metrics = ListField(allow_empty=False)
    postprocessing = ListField(allow_empty=False, optional=True)
    preprocessing = ListField(allow_empty=False, optional=True)
    reader = DataReaderField(optional=True)
    annotation_conversion = DictField(optional=True)
    subsample_size = BaseField(optional=True)
    subsample_seed = NumberField(floats=False, min_value=0, optional=True)


class Dataset:
    def __init__(self, config_entry, preprocessor):
        self._config = config_entry
        self._preprocessor = preprocessor

        self.batch = 1

        dataset_config = DatasetConfig('Dataset')
        data_reader_config = self._config.get('reader', 'opencv_imread')
        if isinstance(data_reader_config, str):
            self.read_image_fn = BaseReader.provide(data_reader_config)
        elif isinstance(data_reader_config, dict):
            self.read_image_fn = BaseReader.provide(data_reader_config['type'], data_reader_config)
        else:
            raise ConfigError('reader should be dict or string')

        dataset_config.fields['data_source'].is_directory = self.read_image_fn.data_source_is_dir
        dataset_config.fields['data_source'].optional = self.read_image_fn.data_source_optional
        dataset_config.validate(self._config)
        annotation, meta = None, None
        self._images_dir = Path(self._config.get('data_source', ''))
        if 'annotation_conversion' in self._config:
            annotation, meta = self._convert_annotation()
        else:
            stored_annotation = self._config.get('annotation')
            if stored_annotation:
                annotation = read_annotation(get_path(stored_annotation))
                meta = self._load_meta()

        if not annotation:
            raise ConfigError('path to converted annotation or data for conversion should be specified')

        subsample_size = self._config.get('subsample_size')
        if subsample_size:
            subsample_seed = self._config.get('subsample_seed', 666)
            if isinstance(subsample_size, str):
                if subsample_size.endswith('%'):
                    subsample_size = float(subsample_size[:-1]) / 100 * len(annotation)
            subsample_size = int(subsample_size)
            annotation = make_subset(annotation, subsample_size, subsample_seed)

        if contains_all(self._config, ['annotation', 'annotation_conversion']):
            annotation_name = self._config['annotation']
            meta_name = self._config.get('dataset_meta')
            if meta_name:
                meta_name = Path(meta_name)
            save_annotation(annotation, meta, Path(annotation_name), meta_name)

        self._annotation = annotation
        self._meta = meta
        self.size = len(self._annotation)
        self.name = self._config.get('name')

    @property
    def annotation(self):
        return self._annotation

    def __len__(self):
        return self.size

    @property
    def metadata(self):
        return self._meta

    @property
    def labels(self):
        return self._meta.get('label_map', {})

    def __getitem__(self, item):
        if self.size <= item * self.batch:
            raise IndexError

        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_annotation = self._annotation[batch_start:batch_end]

        identifiers = [annotation.identifier for annotation in batch_annotation]
        images = self._read_images(identifiers)

        for image, annotation in zip(images, batch_annotation):
            self.set_annotation_metadata(annotation, image)

        preprocessed = self._preprocessor.process(images, batch_annotation)

        return batch_annotation, preprocessed

    @staticmethod
    def set_image_metadata(annotation, images):
        image_sizes = []
        if not isinstance(images, list):
            images = [images]
        for image in images:
            if np.isscalar(image):
                image_sizes.append((1,))
            else:
                image_sizes.append(image.shape)
        annotation.set_image_size(image_sizes)

    def set_annotation_metadata(self, annotation, image):
        self.set_image_metadata(annotation, image.data)
        annotation.set_data_source(self._images_dir)

    def _read_images(self, identifiers):
        images = []
        for identifier in identifiers:
            images.append(DataRepresentation(self.read_image_fn(identifier, self._images_dir), identifier=identifier))

        return images

    def _load_meta(self):
        meta_data_file = self._config.get('dataset_meta')
        return read_json(meta_data_file, cls=JSONDecoderWithAutoConversion) if meta_data_file else None

    def _convert_annotation(self):
        conversion_params = self._config.get('annotation_conversion')
        converter = conversion_params['converter']
        annotation_converter = BaseFormatConverter.provide(converter, conversion_params)
        annotation, meta = annotation_converter.convert()

        return annotation, meta


def read_annotation(annotation_file: Path):
    annotation_file = get_path(annotation_file)

    result = []
    with annotation_file.open('rb') as file:
        while True:
            try:
                result.append(BaseRepresentation.load(file))
            except EOFError:
                break

    return result
