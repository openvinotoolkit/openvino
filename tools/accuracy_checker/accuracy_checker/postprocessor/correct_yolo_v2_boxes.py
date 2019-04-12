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

from ..config import NumberField
from .postprocessor import BasePostprocessorConfig, Postprocessor
from ..representation import DetectionPrediction, DetectionAnnotation
from ..utils import get_size_from_config


class CorrectYoloV2Boxes(Postprocessor):
    __provider__ = 'correct_yolo_v2_boxes'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    def validate_config(self):
        class _CorrectYoloV2BoxesConfigValidator(BasePostprocessorConfig):
            dst_width = NumberField(floats=False, optional=True, min_value=1)
            dst_height = NumberField(floats=False, optional=True, min_value=1)
            size = NumberField(floats=False, optional=True, min_value=1)

        clip_config_validator = _CorrectYoloV2BoxesConfigValidator(
            self.__provider__, on_extra_argument=_CorrectYoloV2BoxesConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        clip_config_validator.validate(self.config)

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process_image(self, annotation, prediction):
        dst_h, dst_w = self.dst_height, self.dst_width
        # postprocessor always expects lists of annotations and predictions for the same image
        # we do not need to get image sizes in cycle, because they are equal
        img_h, img_w, _ = self.image_size

        if (dst_w / img_w) < (dst_h / img_h):
            new_w = dst_w
            new_h = (img_h * dst_w) // img_w
        else:
            new_h = dst_h
            new_w = (img_w * dst_h) // img_h

        for prediction_ in prediction:
            coordinates = zip(prediction_.x_mins, prediction_.y_mins, prediction_.x_maxs, prediction_.y_maxs)
            for i, (x0, y0, x1, y1) in enumerate(coordinates):
                box = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]
                box[0] = (box[0] - (dst_w - new_w) / (2.0 * dst_w)) * (dst_w / new_w)
                box[1] = (box[1] - (dst_h - new_h) / (2.0 * dst_h)) * (dst_h / new_h)
                box[2] *= dst_w / new_w
                box[3] *= dst_h / new_h

                box[0] *= img_w
                box[1] *= img_h
                box[2] *= img_w
                box[3] *= img_h

                prediction_.x_mins[i] = box[0] - box[2] / 2.0 + 1
                prediction_.y_mins[i] = box[1] - box[3] / 2.0 + 1
                prediction_.x_maxs[i] = box[0] + box[2] / 2.0 + 1
                prediction_.y_maxs[i] = box[1] + box[3] / 2.0 + 1

        return annotation, prediction
