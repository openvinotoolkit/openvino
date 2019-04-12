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

import numpy as np


class AverageMeter:
    def __init__(self, loss=None, counter=None):
        self.loss = loss or (lambda x, y: int(x == y))
        self.counter = counter or (lambda x: 1)
        self.accumulator = None
        self.total_count = None

    def update(self, annotation_val, prediction_val):
        loss = self.loss(annotation_val, prediction_val)
        increment = self.counter(annotation_val)

        if self.accumulator is None and self.total_count is None:
            # wrap in array for using numpy.divide with where attribute
            # and support cases when loss function returns list-like object
            self.accumulator = np.array(loss, dtype=float)
            self.total_count = np.array(increment, dtype=float)
        else:
            self.accumulator += loss
            self.total_count += increment

    def evaluate(self):
        if self.total_count is None:
            return 0.0

        return np.divide(
            self.accumulator, self.total_count, out=np.zeros_like(self.accumulator), where=self.total_count != 0
        )
