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

from ..config import BaseField, ConfigValidator, StringField
from ..dependency import ClassProvider


class Adapter(ClassProvider):
    """
    Interface that describes converting raw output to appropriate representation.
    """

    __provider_type__ = 'adapter'

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        self.launcher_config = launcher_config
        self.output_blob = output_blob
        self.label_map = label_map

        self.validate_config()
        self.configure()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process(self, raw, identifiers=None, frame_meta=None):
        raise NotImplementedError

    def configure(self):
        pass

    def validate_config(self):
        pass

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        return outputs_list[0]


class AdapterField(BaseField):
    def validate(self, entry, field_uri_=None):
        super().validate(entry, field_uri_)

        if entry is None:
            return

        field_uri_ = field_uri_ or self.field_uri
        if isinstance(entry, str):
            StringField(choices=Adapter.providers).validate(entry, 'adapter')
        elif isinstance(entry, dict):
            class DictAdapterValidator(ConfigValidator):
                type = StringField(choices=Adapter.providers)
            dict_adapter_validator = DictAdapterValidator(
                'adapter', on_extra_argument=DictAdapterValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            dict_adapter_validator.validate(entry)
        else:
            self.raise_error(entry, field_uri_, 'adapter must be either string or dictionary')
