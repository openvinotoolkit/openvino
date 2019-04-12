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

from ..config import ConfigValidator, StringField
from ..utils import overrides, zipped_transform
from .postprocessor import Postprocessor


class PostprocessingExecutor:
    def __init__(self, processors=None, dataset_name='custom', dataset_meta=None, state=None):
        self._processors = []
        self._image_processors = []
        self._dataset_processors = []
        self.dataset_meta = dataset_meta

        self.state = state or {}

        if not processors:
            return

        for config in processors:
            postprocessor_config = PostprocessorConfig(
                "{}.postprocessing".format(dataset_name),
                on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            postprocessor_config.validate(config)
            postprocessor = Postprocessor.provide(config['type'], config, config['type'], self.dataset_meta, state)
            self._processors.append(postprocessor)

        allow_image_postprocessor = True
        for processor in self._processors:
            if overrides(processor, 'process_all', Postprocessor):
                allow_image_postprocessor = False
                self._dataset_processors.append(processor)
            else:
                if allow_image_postprocessor:
                    self._image_processors.append(processor)
                else:
                    self._dataset_processors.append(processor)

    def process_dataset(self, annotations, predictions):
        for method in self._dataset_processors:
            annotations, predictions = method.process_all(annotations, predictions)

        return annotations, predictions

    def process_image(self, annotation, prediction):
        for method in self._image_processors:
            annotation_entries, prediction_entries = method.get_entries(annotation, prediction)
            method.process(annotation_entries, prediction_entries)

        return annotation, prediction

    def process_batch(self, annotations, predictions):
        return zipped_transform(self.process_image, annotations, predictions)

    def full_process(self, annotations, predictions):
        return self.process_dataset(*self.process_batch(annotations, predictions))

    @property
    def has_dataset_processors(self):
        return len(self._dataset_processors) != 0


class PostprocessorConfig(ConfigValidator):
    type = StringField(choices=Postprocessor.providers)
