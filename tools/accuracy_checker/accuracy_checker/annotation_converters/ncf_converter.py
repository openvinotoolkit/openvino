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


from ..representation import HitRatioAnnotation
from ..utils import read_txt
from ..config import PathField, NumberField

from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class NCFDatasetConverterConfig(BaseFormatConverterConfig):
    raiting_file = PathField()
    negative_file = PathField()
    users_max_number = NumberField(optional=True)


class NCFConverter(BaseFormatConverter):
    __provider__ = "ncf_converter"

    _config_validator_type = NCFDatasetConverterConfig

    def configure(self):
        self.raiting_file = self.config['raiting_file']
        self.negative_file = self.config['negative_file']
        if 'users_max_number' in self.config:
            self.users_max_number = self.config['users_max_number']
        else:
            self.users_max_number = -1

    def convert(self):
        annotations = []
        users = []

        for file_row in read_txt(self.raiting_file):
            user_id, item_id, _ = file_row.split()
            users.append(user_id)
            identifier = ['u:'+user_id, 'i:'+item_id]
            annotations.append(HitRatioAnnotation(identifier))
            if self.users_max_number > 0 and len(users) >= self.users_max_number:
                break;

        item_numbers = 1

        items_neg = []
        for file_row in read_txt(self.negative_file):
            items = file_row.split()
            items_neg.append(items)
            if self.users_max_number > 0 and len(items_neg) >= self.users_max_number:
                break;

        if items_neg:
            iterations = len(items_neg[0])
            item_numbers += iterations
            for i in range(iterations):
                for user in users:
                    item = items_neg[int(user)][i]
                    identifier = ['u:' + user, 'i:'+ item]
                    annotations.append(HitRatioAnnotation(identifier, False))

        return annotations, {'users_number': len(users), 'item_numbers': item_numbers}
