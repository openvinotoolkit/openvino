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

from functools import singledispatch
from collections import OrderedDict
import re
import cv2
from PIL import Image
import scipy.misc
import numpy as np
import nibabel as nib

from ..utils import get_path, read_json
from ..dependency import ClassProvider
from ..config import BaseField, StringField, ConfigValidator, ConfigError, DictField


class DataReaderField(BaseField):
    def validate(self, entry_, field_uri=None):
        super().validate(entry_, field_uri)

        if entry_ is None:
            return

        field_uri = field_uri or self.field_uri
        if isinstance(entry_, str):
            StringField(choices=BaseReader.providers).validate(entry_, 'reader')
        elif isinstance(entry_, dict):
            class DictReaderValidator(ConfigValidator):
                type = StringField(choices=BaseReader.providers)
            dict_reader_validator = DictReaderValidator(
                'reader', on_extra_argument=DictReaderValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            dict_reader_validator.validate(entry_)
        else:
            self.raise_error(entry_, field_uri, 'reader must be either string or dictionary')


class BaseReader(ClassProvider):
    __provider_type__ = 'reader'

    def __init__(self, config=None):
        self.config = config
        self.data_source_is_dir = True
        self.data_source_optional = False
        self.read_dispatcher = singledispatch(self.read)
        self.read_dispatcher.register(list, self._read_list)

        self.validate_config()
        self.configure()

    def __call__(self, *args, **kwargs):
        return self.read_dispatcher(*args, **kwargs)

    def configure(self):
        pass

    def validate_config(self):
        pass

    def read(self, data_id, data_dir):
        raise NotImplementedError

    def _read_list(self, data_id, data_dir):
        return [self.read(identifier, data_dir) for identifier in data_id]


class ReaderCombinerConfig(ConfigValidator):
    type = StringField()
    scheme = DictField(
        value_type=DataReaderField(), key_type=StringField(), allow_empty=False
    )


class ReaderCombiner(BaseReader):
    __provider__ = 'combine_reader'

    def validate_config(self):
        config_validator = ReaderCombinerConfig('reader_combiner_config')
        config_validator.validate(self.config)

    def configure(self):
        scheme = self.config['scheme']
        reading_scheme = OrderedDict()
        for pattern, reader_config in scheme.items():
            reader = BaseReader.provide(
                reader_config['type'] if isinstance(reader_config, dict) else reader_config, reader_config
            )
            pattern = re.compile(pattern)
            reading_scheme[pattern] = reader

        self.reading_scheme = reading_scheme

    def read(self, data_id, data_dir):
        for pattern, reader in self.reading_scheme.items():
            if pattern.match(str(data_id)):
                return reader.read(data_id, data_dir)

        raise ConfigError('suitable data reader for {} not found'.format(data_id))


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    def read(self, data_id, data_dir):
        return cv2.imread(str(get_path(data_dir / data_id)))


class PillowImageReader(BaseReader):
    __provider__ = 'pillow_imread'

    def read(self, data_id, data_dir):
        return np.array(Image.open(str(get_path(data_dir / data_id))))


class ScipyImageReader(BaseReader):
    __provider__ = 'scipy_imread'

    def read(self, data_id, data_dir):
        return np.array(scipy.misc.imread(str(get_path(data_dir / data_id))))

class OpenCVFrameReader(BaseReader):
    __provider__ = 'opencv_capture'

    def __init__(self, config=None):
        super().__init__(config)
        self.data_source_is_dir = False
        self.source = None
        self.current = -1

    def read(self, data_id, data_dir):
        # source video changed, capture initialization
        if data_dir != self.source:
            self.source = data_dir
            self.videocap = cv2.VideoCapture(str(self.source))
            self.current = -1

        if data_id < 0:
            raise IndexError('frame with {} index can not be grabbed, non-negative index is expected')
        if data_id < self.current:
            self.videocap.set(cv2.CAP_PROP_POS_FRAMES, data_id)
            self.current = data_id - 1

        return self._read_sequence(data_id)

    def _read_sequence(self, data_id):
        frame = None
        while self.current != data_id:
            success, frame = self.videocap.read()
            self.current += 1
            if not success:
                raise EOFError('frame with {} index does not exists in {}'.format(self.current, self.source))
        return frame


class JSONReaderConfig(ConfigValidator):
    type = StringField()
    key = StringField(optional=True, case_sensitive=True)


class JSONReader(BaseReader):
    __provider__ = 'json_reader'

    def validate_config(self):
        config_validator = JSONReaderConfig('json_reader_config')
        config_validator.validate(self.config)

    def configure(self):
        self.key = self.config.get('key')

    def read(self, data_id, data_dir):
        data = read_json(str(data_dir / data_id))
        if self.key:
            data = data.get(self.key)

            if not data:
                raise ConfigError('{} does not contain {}'.format(data_id, self.key))

        return np.array(data).astype(np.float32)

class NCF_DataReader(BaseReader):
    __provider__ = 'ncf_data_reader'

    def __init__(self, config=None):
        super().__init__(config)
        self.data_source_optional = True

    def read(self, data_id, data_dir):
        if not isinstance(data_id, str):
            raise IndexError('Data identifier must be a string')

        return float(data_id.split(":")[1])

class NiftiImageReader(BaseReader):
    __provider__ = 'nifti_reader'

    def read(self, data_id, data_dir):
        nib_image = nib.load(str(get_path(data_dir / data_id)))
        image = np.array(nib_image.dataobj)
        if len(image.shape) != 4:  # Make sure 4D
            image = np.expand_dims(image, -1)
        image = np.swapaxes(np.array(image), 0, -2)
        return image
