"""
Copyright (C) 2018-2019 Intel Corporation

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


class TopResults:
    def __init__(self, data, channels_count: int):
        self.__results = list()

        samples = int(data.size / channels_count)
        for sample in range(samples):
            max_value = None
            max_value_class_number = None

            for class_number in range(channels_count):
                value = data.item(class_number + sample * channels_count)
                if (max_value is None) or (max_value < value):
                    max_value = value
                    max_value_class_number = class_number

            self.__results.append(max_value_class_number)

    @property
    def results(self):
        return self.__results
