# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ing templates processed by testing framework.
"""
from collections import OrderedDict

from e2e_tests.common.decorators import wrap_ord_dict


@wrap_ord_dict
def squeeze(axis=(2, 3)):
    """Construct squeezing action.

    :param axis: axis along which squeezing is performed, defaults to (2, 3)
    :return: "squeeze" action processed by testing framework
    """
    return "squeeze", {"axis": axis}


@wrap_ord_dict
def parse_object_detection():
    """Construct object detection parsing action."""
    return "parse_object_detection", {}


def align_with_batch_postprocess():
    """Construct align_with_batch postprocess."""
    return "parse_object_detection", {}


@wrap_ord_dict
def parse_semantic_segmentation():
    """Construct object detection parsing action."""
    return "parse_semantic_segmentation", {}


@wrap_ord_dict
def parse_image_modification():
    """Construct object detection parsing action."""
    return "parse_image_modification", {}


@wrap_ord_dict
def parse_classification(labels_offset=0, target_layers=None):
    """Construct classification parsing action.

    :param labels_offset: offset to be used during results parsing. i.e.
                          imagenet classification model can return 1001 class
                          where class 0 represents "background", one can
                          specify labels_offset=1 in order to cut the
                          "background" class (1001 -> 1000), defaults to 0
    :return: "parse_classification" action processed by testing framework.
    """
    return "parse_classification", {"labels_offset": labels_offset, 'target_layers': target_layers}


@wrap_ord_dict
def squeeze_and_parse_classification(axis=(2, 3), labels_offset=0):
    """Construct squeeze and parse classification actions in a single pipeline.

    :param axis: axis along which squeezing is performed, defaults to (2, 3)
    :param labels_offset: offset to be used during results parsing. i.e.
                          imagenet classification model can return 1001 class
                          where class 0 represents "background", one can
                          specify labels_offset=1 in order to cut the
                          "background" class (1001 -> 1000), defaults to 0
    :return: "squeeze" and "parse_classification" action processed by testing
             framework
    """
    return [squeeze.unwrap(axis=axis), parse_classification.unwrap(labels_offset=labels_offset)]


def assemble_postproc_tf(batch=None, align_with_batch_od=False, **kwargs):
    """Add mxnet-specific postprocessing. Pass rest of arguments as is.

    :return: "postprocess" step with MXNet specific actions
    """
    steps = []
    if batch is not None and align_with_batch_od:
        steps.append(("align_with_batch_od", {"batch": batch}))
    elif batch is not None:
        steps.append(("align_with_batch", {"batch": batch}))
    for preproc, config in kwargs.items():
        steps.append((preproc, config))

    return "postprocessor", dict(steps)


def paddlepaddle_od_postproc(target_layers=None):
    """Construct PaddlePaddle object detection parsing actions."""
    if target_layers is None:
        target_layers = {}
    return OrderedDict([
        ("mxnet_to_common_od_format", {"target_layers": target_layers}),  # PDPD has the same OD format as MXNET
        ("parse_object_detection", {"target_layers": target_layers})
    ])


@wrap_ord_dict
def parse_image_modification():
    """Construct object detection parsing action."""
    return "parse_image_modification", {}
