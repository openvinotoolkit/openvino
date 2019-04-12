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

import pytest
import numpy as np
from accuracy_checker.metrics import DetectionMAP
from accuracy_checker.metrics.detection import Recall, bbox_match
from accuracy_checker.metrics.overlap import IOU, IOA
from tests.common import (make_representation, single_class_dataset, multi_class_dataset,
                          multi_class_dataset_without_background)


def _test_metric_wrapper(metric_cls, dataset, **kwargs):
    provider = metric_cls.__provider__
    config = {'type': provider, 'name': provider}
    config.update(**kwargs)
    return metric_cls(config, dataset, provider)


class TestBoxMatch:
    def test_single(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert fp[0] == 0

    def test_single_with_ignored_tp(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        pred[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 0
        assert fp[0] == 0

    def test_single_with_use_filtered_tp(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        pred[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator, use_filtered_tp=True)
        assert tp[0] == 1
        assert fp[0] == 0

    def test_single_non_overlap(self):
        gt = make_representation("0 5 5 10 10", is_ground_truth=True)
        pred = make_representation("0 0 0 5 5", score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 0
        assert fp[0] == 1

    def test_single_non_overlap_ignored(self):
        gt = make_representation("0 5 5 10 10", is_ground_truth=True)
        pred = make_representation("0 0 0 5 5", score=1)
        pred[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 0
        assert fp[0] == 0

    def test_multiple(self):
        gt = make_representation("0 0 0 5 5; 0 7 7 8 8", is_ground_truth=True)
        pred = make_representation("0 0 0 5 5; 0 7 7 8 8", score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert tp[1] == 1
        assert fp[0] == 0
        assert fp[0] == 0

    def test_multiple_2(self):
        gt = make_representation("0 0 0 5 5; 0 9 9 10 10", is_ground_truth=True)
        pred = make_representation("1 0 0 0 5 5; 0.8 0 7 7 8 8")
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert tp[1] == 0
        assert fp[0] == 0
        assert fp[1] == 1

    def test_multi_label(self):
        gt = make_representation("1 0 0 5 5; 0 9 9 10 10", is_ground_truth=True)
        pred = make_representation("1 1 0 0 5 5; 0.8 0 7 7 8 8")
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 1, overlap_evaluator)
        assert tp.shape[0] == 1
        assert tp[0] == 1
        assert fp[0] == 0

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp.shape[0] == 1
        assert tp[0] == 0
        assert fp[0] == 1

    def test_multi_image(self):
        gt = make_representation(["0 0 0 5 5", "0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5", "0 0 0 5 5"], score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert tp[1] == 1
        assert fp[0] == 0
        assert fp[1] == 0

    def test_false_negative(self):
        gt = make_representation("0 0 0 5 5; 0 1 1 6 6", is_ground_truth=True)
        pred = make_representation("0 0 0 5 5", score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, ngt = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert tp.shape[0] == 1
        assert ngt == 2

    def test_multiple_detections(self):
        gt = make_representation("0 0 0 5 5", is_ground_truth=True)
        pred = make_representation("1 0 0 0 5 5; 0.9 0 0 0 5 5")
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 1
        assert tp[1] == 0

    def test_no_annotations(self):
        gt = "1 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, _ = bbox_match(gt, pred, 0, overlap_evaluator)
        assert tp[0] == 0
        assert fp[0] == 1

    def test_no_predictions(self):
        gt = "0 0 0 5 5"
        pred = "1 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator)
        assert n == 1
        assert len(tp) == 0
        assert len(fp) == 0

    def test_iou_empty_prediction_box(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 0 0"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOU({})

        with pytest.warns(None) as warnings:
            tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator)
            assert len(warnings) == 0
            assert n == 1
            assert tp[0] == 0
            assert fp[0] == 1

    def test_ioa_empty_prediction_box(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 0 0"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOA({})

        with pytest.warns(None) as warnings:
            tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator)
            assert len(warnings) == 0
            assert n == 1
            assert tp[0] == 0
            assert fp[0] == 1

    def test_iou_zero_union(self):
        gt = "0 0 0 0 0"
        pred = "0 0 0 0 0"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        overlap_evaluator = IOA({})

        with pytest.warns(None) as warnings:
            tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator)
            assert len(warnings) == 0
            assert n == 1
            assert tp[0] == 0
            assert fp[0] == 1

    def test_single_difficult(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        gt[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator, ignore_difficult=True)
        assert n == 0
        assert tp[0] == 0
        assert fp[0] == 0

    def test_single_with_not_ignore_difficult(self):
        gt = "0 0 0 5 5"
        pred = "0 0 0 5 5"

        gt = make_representation(gt, is_ground_truth=True)
        pred = make_representation(pred, score=1)
        gt[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator, ignore_difficult=False)
        assert n == 1
        assert tp[0] == 1
        assert fp[0] == 0

    def test_single_difficult_non_overlap(self):
        gt = make_representation("0 5 5 10 10", is_ground_truth=True)
        gt[0].metadata['difficult_boxes'] = [0]
        pred = make_representation("0 0 0 5 5", score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator)
        assert n == 0
        assert tp[0] == 0
        assert fp[0] == 1

    def test_single_difficult_non_overlap_not_ignore_difficult(self):
        gt = make_representation("0 5 5 10 10", is_ground_truth=True)
        gt[0].metadata['difficult_boxes'] = [0]
        pred = make_representation("0 0 0 5 5", score=1)
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator, ignore_difficult=False)
        assert n == 1
        assert tp[0] == 0
        assert fp[0] == 1

    def test_multiple_detections_with_ignore_difficult(self):
        gt = make_representation("0 0 0 5 5", is_ground_truth=True)
        pred = make_representation("1 0 0 0 5 5; 0.9 0 0 0 5 5")
        gt[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator, ignore_difficult=True)
        assert n == 0
        assert tp[0] == 0
        assert tp[1] == 0
        assert fp[0] == 0
        assert fp[1] == 0

    def test_multiple_detections_with_not_ignore_difficult(self):
        gt = make_representation("0 0 0 5 5", is_ground_truth=True)
        pred = make_representation("1 0 0 0 5 5; 0.9 0 0 0 5 5")
        gt[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(gt, pred, 0, overlap_evaluator, ignore_difficult=False)
        assert n == 1
        assert tp[0] == 1
        assert tp[1] == 0
        assert fp[0] == 0
        assert fp[1] == 1

    def test_multiple_detections_with_ignore_difficult_and_not_allow_multiple_matches_per_ignored(self):
        gt = make_representation("0 0 0 5 5", is_ground_truth=True)
        pred = make_representation("1 0 0 0 5 5; 0.9 0 0 0 5 5")
        gt[0].metadata['difficult_boxes'] = [0]
        overlap_evaluator = IOU({})

        tp, fp, _, n = bbox_match(
            gt, pred, 0, overlap_evaluator,
            ignore_difficult=True, allow_multiple_matches_per_ignored=False
        )

        assert n == 0
        assert tp[0] == 0
        assert tp[1] == 0
        assert fp[0] == 0
        assert fp[1] == 1


class TestRecall:
    def test_one_object(self):
        gt = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5"], score=1)
        metric = _test_metric_wrapper(Recall, single_class_dataset())
        assert 1 == metric(gt, pred)[0]
        assert metric.meta.get('names') == ['dog']

    def test_two_objects(self):
        gt = make_representation(["0 0 0 5 5; 0 10 10 20 20"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 0 10 10 20 20"], score=1)
        assert 1 == _test_metric_wrapper(Recall, single_class_dataset())(gt, pred)[0]

    def test_false_positive(self):
        gt2 = make_representation(["0 10 10 20 20"], is_ground_truth=True)
        pred2 = make_representation(["0 0 0 5 5"], score=1)
        metric = _test_metric_wrapper(Recall, single_class_dataset())
        assert 0 == metric(gt2, pred2)[0]
        assert metric.meta.get('names') == ['dog']

        gt1 = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred1 = make_representation(["0 0 0 5 5; 0 10 10 20 20"], score=1)
        assert 1 == metric(gt1, pred1)[0]
        assert metric.meta.get('names') == ['dog']

    def test_false_negative(self):
        gt = make_representation(["0 10 10 20 20; 0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5"], score=1)
        metric = _test_metric_wrapper(Recall, single_class_dataset())
        assert 0.5 == metric(gt, pred)[0]
        assert metric.meta.get('names') == ['dog']

    def test_duplicate_detections(self):
        gt = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 0 0 0 5 5"], score=1)

        metric = _test_metric_wrapper(Recall, single_class_dataset())
        assert 1 == metric(gt, pred)[0]
        assert metric.meta.get('names') == ['dog']

    def test_no_warnings_in_recall_calculation(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], score=1)

        with pytest.warns(None) as warnings:
            _test_metric_wrapper(Recall, multi_class_dataset())(gt, pred)
        assert len(warnings) == 0

    def test_on_dataset_without_background(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], score=1)

        with pytest.warns(None) as warnings:
            _test_metric_wrapper(Recall, multi_class_dataset_without_background())(gt, pred)
        assert len(warnings) == 0

    def test_not_gt_boxes_for_matching(self):
        gt = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["1 0 0 5 5"], score=1)

        metric = _test_metric_wrapper(Recall, multi_class_dataset_without_background())
        assert 0 == metric(gt, pred)[0]
        assert metric.meta.get('names') == ['cat']


class TestMAP:
    def test_selects_all_detections(self):
        gt = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 0 0 0 5 5"], score=1)

        metric = _test_metric_wrapper(DetectionMAP, single_class_dataset())
        metric(gt, pred)

        assert not metric.distinct_conf
        assert metric.overlap_threshold == 0.5
        assert metric.ignore_difficult
        assert metric.meta.get('names') == ['dog']

    def test_no_warnings_in_map_calculation(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], score=1)

        with pytest.warns(None) as warnings:
            _test_metric_wrapper(DetectionMAP, multi_class_dataset())(gt, pred)
        assert len(warnings) == 0

    def test_perfect_detection(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], score=1)

        metric = _test_metric_wrapper(DetectionMAP, multi_class_dataset())
        assert metric(gt, pred) == [1.0, 1.0]
        assert metric.meta.get('names') == ['dog', 'cat']

    def test_one_false_alarm(self):
        gt = make_representation(["0 0 0 5 5", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["1 10 10 20 20; 0 0 0 5 5", "1 0 0 5 5"], score=1)
        metric = _test_metric_wrapper(DetectionMAP, multi_class_dataset())
        values = metric(gt, pred)
        assert values == [1.0, 0.5]
        map_ = np.mean(values)
        assert 0.75 == map_
        assert metric.meta.get('names') == ['dog', 'cat']

    def test_zero_detection(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20"], is_ground_truth=True)
        pred = make_representation(["0 30 30 40 40"], score=1)

        metric = _test_metric_wrapper(DetectionMAP, multi_class_dataset())
        assert metric(gt, pred) == [0.0]
        assert metric.meta.get('names') == ['dog']

    def test_no_detections_warn_user_warning(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20"], is_ground_truth=True)
        pred = make_representation("", score=1)
        with pytest.warns(UserWarning) as warnings:
            map_ = _test_metric_wrapper(DetectionMAP, multi_class_dataset())(gt, pred)[0]
            assert len(warnings) == 1

            assert map_ == 0

    def test_detection_on_dataset_without_background(self):
        gt = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["0 0 0 5 5; 1 10 10 20 20", "1 0 0 5 5"], score=1)

        with pytest.warns(None) as warnings:
            map_ = _test_metric_wrapper(DetectionMAP, multi_class_dataset_without_background())(gt, pred)
            mean = np.mean(map_)
            assert 1.0 == mean
        assert len(warnings) == 0

    def test_not_gt_boxes_for_box_matching(self):
        gt = make_representation(["0 0 0 5 5"], is_ground_truth=True)
        pred = make_representation(["1 0 0 5 5"], score=1)

        metric = _test_metric_wrapper(Recall, multi_class_dataset_without_background())
        assert 0 == metric(gt, pred)[0]
        assert metric.meta.get('names') == ['cat']
