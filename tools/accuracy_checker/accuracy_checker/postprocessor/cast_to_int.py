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
from typing import Union
import numpy as np
from ..config import StringField
from ..representation import DetectionAnnotation, DetectionPrediction, TextDetectionPrediction, TextDetectionAnnotation
from .postprocessor import Postprocessor, BasePostprocessorConfig

round_policies_func = {
    'nearest': np.rint,
    'nearest_to_zero': np.trunc,
    'lower': np.floor,
    'greater': np.ceil
}


class CastToIntConfigValidator(BasePostprocessorConfig):
    round_policy = StringField(optional=True, choices=round_policies_func.keys())


class CastToInt(Postprocessor):
    __provider__ = 'cast_to_int'
    annotation_types = (DetectionAnnotation, TextDetectionAnnotation)
    prediction_types = (DetectionPrediction, TextDetectionPrediction)
    _config_validator_type = CastToIntConfigValidator

    def configure(self):
        self.round_func = round_policies_func[self.config.get('round_policy', 'nearest')]

    def process_image(self, annotation, prediction):
        @singledispatch
        def cast(entry):
            pass

        @cast.register(Union[DetectionAnnotation, DetectionPrediction])
        def _(entry):
            entry.x_mins = self.round_func(entry.x_mins)
            entry.x_maxs = self.round_func(entry.x_maxs)
            entry.y_mins = self.round_func(entry.y_mins)
            entry.y_maxs = self.round_func(entry.y_maxs)

        @cast.register(Union[TextDetectionAnnotation, TextDetectionPrediction])
        def _(entry):
            entry.points = self.round_func(entry.points)


        for annotation_ in annotation:
            cast(annotation_)

        for prediction_ in prediction:
            cast(prediction_)

        return annotation, prediction
