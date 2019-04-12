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

from .base_representation import BaseRepresentation


class ReIdentification(BaseRepresentation):
    pass


class ReIdentificationAnnotation(ReIdentification):
    def __init__(self, identifier, camera_id, person_id, query):
        super().__init__(identifier)
        self.camera_id = camera_id
        self.person_id = person_id
        self.query = query


class ReIdentificationClassificationAnnotation(ReIdentification):
    def __init__(self, identifier, positive_pairs=None, negative_pairs=None):
        super().__init__(identifier)
        self.positive_pairs = set(positive_pairs)
        self.negative_pairs = set(negative_pairs)


class ReIdentificationPrediction(ReIdentification):
    def __init__(self, identifiers, embedding):
        super().__init__(identifiers)
        self.embedding = embedding.copy()
