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

from .annotation_converters import BaseFormatConverter, save_annotation, make_subset
from .config import ConfigValidator, StringField, PathField, ListField, DictField, BaseField, NumberField, ConfigError
from .utils import JSONDecoderWithAutoConversion, read_json, get_path, contains_all
from .representation import BaseRepresentation
from .data_readers import DataReaderField


class DatasetConfig(ConfigValidator):
    """
    Specifies configuration structure for dataset
    """
    name = StringField()
    annotation = PathField(optional=True, check_exists=False)
    data_source = PathField(optional=True, check_exists=False)
    dataset_meta = PathField(optional=True, check_exists=False)
    metrics = ListField(allow_empty=False, optional=True)
    postprocessing = ListField(allow_empty=False, optional=True)
    preprocessing = ListField(allow_empty=False, optional=True)
    reader = DataReaderField(optional=True)
    annotation_conversion = DictField(optional=True)
    subsample_size = BaseField(optional=True)
    subsample_seed = NumberField(floats=False, min_value=0, optional=True)


class Dataset:
    def __init__(self, config_entry):
        self._config = config_entry
        self.batch = 1
        self.iteration = 0
        dataset_config = DatasetConfig('Dataset')
        dataset_config.validate(self._config)
        annotation, meta = None, None
        use_converted_annotation = True
        self._images_dir = Path(self._config.get('data_source', ''))
        if 'annotation' in self._config:
            annotation_file = Path(self._config['annotation'])
            if annotation_file.exists():
                annotation = read_annotation(get_path(annotation_file))
                meta = self._load_meta()
                use_converted_annotation = False
        if not annotation and 'annotation_conversion' in self._config:
            annotation, meta = self._convert_annotation()

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

        if use_converted_annotation and contains_all(self._config, ['annotation', 'annotation_conversion']):
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

    def __call__(self, context, *args, **kwargs):
        batch_annotation = self.__getitem__(self.iteration)
        self.iteration += 1
        context.annotation_batch = batch_annotation
        context.identifiers_batch = [annotation.identifier for annotation in batch_annotation]

    def __getitem__(self, item):
        if self.size <= item * self.batch:
            raise IndexError

        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_annotation = self._annotation[batch_start:batch_end]

        return batch_annotation

    @staticmethod
    def set_image_metadata(annotation, images):
        image_sizes = []
        data = images.data
        if not isinstance(data, list):
            data = [data]
        for image in data:
            image_sizes.append(image.shape)
        annotation.set_image_size(image_sizes)

    def set_annotation_metadata(self, annotation, image, data_source):
        self.set_image_metadata(annotation, image.data)
        annotation.set_data_source(data_source)

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
