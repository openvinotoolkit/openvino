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
from ..config import PathField, StringField, BoolField
from ..representation import SuperResolutionAnnotation
from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class SRConverterConfig(BaseFormatConverterConfig):
    data_dir = PathField(is_directory=True)
    lr_suffix = StringField(optional=True)
    hr_suffix = StringField(optional=True)
    two_streams = BoolField(optional=True)


class SRConverter(BaseFormatConverter):
    __provider__ = 'super_resolution'

    _config_validator_type = SRConverterConfig

    def configure(self):
        self.data_dir = self.config['data_dir']
        self.lr_suffix = self.config.get('lr_suffix', 'lr')
        self.hr_suffix = self.config.get('hr_suffix', 'hr')
        self.two_streams = self.config.get('two_streams', False)

    def convert(self):
        file_list_lr = []
        for file_in_dir in self.data_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1]:
                file_list_lr.append(file_in_dir)

        annotation = []
        for lr_file in file_list_lr:
            lr_file_name = lr_file.parts[-1]
            hr_file_name = self.hr_suffix.join(lr_file_name.split(self.lr_suffix))
            identifier = [lr_file_name, hr_file_name] if self.two_streams else lr_file_name
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name))

        return annotation, None
