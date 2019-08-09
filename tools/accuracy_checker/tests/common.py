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
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np

from accuracy_checker.representation import DetectionAnnotation, DetectionPrediction, SegmentationPrediction, SegmentationAnnotation
from accuracy_checker.utils import get_path


@contextmanager
# since it seems not possible to create pathlib.Path from str with '/' at the end we accept strings
# expect paths in posix format
def mock_filesystem(hierarchy: List[str]):
    with TemporaryDirectory() as prefix:
        for entry in hierarchy:
            path = Path(prefix) / entry
            if entry.endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                parent = path.parent
                if parent != Path("."):
                    parent.mkdir(parents=True, exist_ok=True)
                # create file
                path.open('w').close()

        yield get_path(prefix, is_directory=True)


def make_representation(bounding_boxes, is_ground_truth=False, score=None, meta=None):
    """
    Args:
        bounding_boxes: string or list of strings `score label x0 y0 x1 y1; label score x0 y0 x1 y1; ...`.
        is_ground_truth: True if bbs are annotation boxes.
        score: value in [0, 1], if not None, all prediction boxes are considered with the given score.
        meta: metadata for representation
    """

    if not isinstance(bounding_boxes, list):
        bounding_boxes = [bounding_boxes]

    result = []
    for idx, box in enumerate(bounding_boxes):
        arr = np.array(np.mat(box))

        if box == "":
            arr = np.array([]).reshape((0, 5))

        if is_ground_truth or score:
            assert arr.shape[1] == 5
        elif not is_ground_truth and not score:
            assert arr.shape[1] == 6

        if not is_ground_truth and score:
            score_ = score
            if np.isscalar(score_) or len(score_) == 1:
                score_ = np.full(arr.shape[0], score_)
            arr = np.c_[score_, arr]

        if is_ground_truth:
            detection = DetectionAnnotation(str(idx), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
        else:
            detection = DetectionPrediction(str(idx), arr[:, 1], arr[:, 0], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5])

        if meta:
            detection.metadata = meta[idx]

        result.append(detection)

    return result


def make_segmentation_representation(mask, ground_truth=False):
    if ground_truth:
        representation = SegmentationAnnotation('identifier', None)
        representation.mask = mask
        return [representation]

    return [SegmentationPrediction('identifier', mask)]


def update_dict(dictionary, **kwargs):
    copied = dictionary.copy()
    copied.update(**kwargs)

    return copied


class DummyDataset:
    def __init__(self, label_map, bg=-1):
        self.label_map = label_map
        self.background = bg
        self.name = 'dummy'

    @property
    def metadata(self):
        return {"label_map": self.label_map, "background_label": self.background}

    @property
    def labels(self):
        return self.metadata['label_map']


def multi_class_dataset():
    labels = {0: 'dog', 1: 'cat', 2: 'human', -1: 'background'}
    return DummyDataset(label_map=labels, bg=-1)


def multi_class_dataset_without_background():
    labels = {0: 'dog', 1: 'cat', 2: 'human'}
    return DummyDataset(label_map=labels)


def single_class_dataset():
    labels = {0: 'dog', -1: 'background'}
    return DummyDataset(label_map=labels, bg=-1)
