"""
Copyright (c) 2018 Intel Corporation

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
from argparse import ArgumentParser

from ..config import ConfigValidator, StringField, PathField
from ..dependency import ClassProvider
from ..utils import format_key


class BaseFormatConverterConfig(ConfigValidator):
    converter = StringField()


class BaseFormatConverter(ClassProvider):
    __provider_type__ = 'converter'

    _config_validator_type = BaseFormatConverterConfig

    @property
    def config_validator(self):
        return self._config_validator_type(
            '{}_converter_config'.format(self.get_name()),
            on_extra_argument=self._config_validator_type.ERROR_ON_EXTRA_ARGUMENT
        )

    def __init__(self, config=None):
        self.config = config
        if config:
            self.validate_config()
            self.configure()

    def convert(self, *args, **kwargs):
        """
        Converts specific annotation format to the ResultRepresentation specific for current dataset/task.

        Returns:
            annotation: list of ResultRepresentations.
            meta: meta-data map for the current dataset.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__provider__

    def get_argparser(self):
        parser = ArgumentParser(add_help=False)
        config_validator = self.config_validator
        fields = config_validator.fields
        for field_name, field in fields.items():
            if field_name == 'converter':
                # it is base argument. Main argparser already use it to get argparser from specific converter.
                # Converter argparser should contain only converter specific arguments.
                continue

            required = not field.optional
            parser.add_argument(
                format_key(field_name), required=required, type=field.type
            )

        return parser

    def validate_config(self):
        self.config_validator.validate(self.config)

    def configure(self):
        pass


class FileBasedAnnotationConverterConfig(BaseFormatConverterConfig):
    annotation_file = PathField()


class FileBasedAnnotationConverter(BaseFormatConverter):
    _config_validator_type = FileBasedAnnotationConverterConfig

    def configure(self):
        self.annotation_file = self.config['annotation_file']

    def convert(self, *args, **kwargs):
        pass


class DirectoryBasedAnnotationConverterConfig(BaseFormatConverterConfig):
    data_dir = PathField(is_directory=True)


class DirectoryBasedAnnotationConverter(BaseFormatConverter):
    _config_validator_type = DirectoryBasedAnnotationConverterConfig

    def configure(self):
        self.data_dir = self.config['data_dir']

    def convert(self, *args, **kwargs):
        pass
