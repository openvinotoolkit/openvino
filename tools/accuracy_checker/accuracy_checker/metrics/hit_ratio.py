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

import heapq
import math

import numpy as np

from ..representation import HitRatioAnnotation, HitRatioPrediction
from .metric import FullDatasetEvaluationMetric, BaseMetricConfig
from ..config import NumberField


class RecommenderConfigValidator(BaseMetricConfig):
    top_k = NumberField(floats=False, min_value=1, optional=True)


class BaseRecommenderMetric(FullDatasetEvaluationMetric):
    annotation_types = (HitRatioAnnotation, )
    prediction_types = (HitRatioPrediction, )
    _config_validator_type = RecommenderConfigValidator

    def __init__(self, discounter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discounter = discounter or (lambda item, rank: int(item in rank))

    def configure(self):
        self.top_k = self.config.get('top_k', 10)
        self.users_num = self.dataset.metadata.get('users_number')
        self.pred_per_user = {i: [] for i in range(self.users_num)}
        self.gt_items = {}

    def update(self, annotation, prediction):
        self.pred_per_user[prediction.user].append((prediction.item, prediction.scores))
        if annotation.positive:
            self.gt_items[annotation.user] = annotation.item

    def evaluate(self, annotations, predictions):
        iter_num = len(self.pred_per_user[0])

        measure = []
        for user in range(self.users_num):
            map_item_score = {}
            for j in range(iter_num):
                item = self.pred_per_user[user][j][0]
                score = self.pred_per_user[user][j][1]
                map_item_score[item] = score
            ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
            measure.append(self.discounter(self.gt_items[user], ranklist))

        return np.mean(measure)


def hit_ratio_discounter(item, rank):
    return int(item in rank)


def ndcg_discounter(item, rank):
    if item in rank:
        return math.log(2) / math.log(rank.index(item) + 2)

    return 0


class HitRatioMetric(BaseRecommenderMetric):
    """
    Class for evaluating Hit Ratio metric
    """

    __provider__ = 'hit_ratio'

    def __init__(self, *args, **kwargs):
        super().__init__(hit_ratio_discounter, *args, **kwargs)


class NDSGMetric(BaseRecommenderMetric):
    """
    Class for evaluating Normalized Discounted Cumulative Gain metric
    """

    __provider__ = 'ndcg'

    def __init__(self, *args, **kwargs):
        super().__init__(ndcg_discounter, *args, **kwargs)
