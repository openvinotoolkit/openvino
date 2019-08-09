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
from ..preprocessor.preprocessors import Preprocessor


class PreprocessingExecutor:
    def __init__(self, processors=None, dataset_name='custom', dataset_meta=None):
        self.processors = []
        self.dataset_meta = dataset_meta

        if not processors:
            return

        identifier = 'type'
        for processor in processors:
            preprocessor_config = PreprocessorConfig(
                "{}.preprocessors".format(dataset_name), on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
            )

            type_ = processor.get(identifier)
            preprocessor_config.validate(processor, type_)
            preprocessor = Preprocessor.provide(processor[identifier], config=processor, name=type_)

            self.processors.append(preprocessor)

    def __call__(self, context, *args, **kwargs):
        batch_data = context.data_batch
        batch_annotation = context.annotation_batch
        context.data_batch = self.process(batch_data, batch_annotation)

    def process(self, images, batch_annotation=None):
        for i, _ in enumerate(images):
            for processor in self.processors:
                images[i] = processor(
                    image=images[i], annotation_meta=batch_annotation[i].metadata if batch_annotation else None
                )

        return images


class PreprocessorConfig(ConfigValidator):
    type = StringField(choices=Preprocessor.providers)
