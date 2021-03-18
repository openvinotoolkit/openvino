# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import warnings
import os

from openvino.inference_engine import IECore, DataPtr

def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

test_net_xml, test_net_bin = model_path()


def layer_out_data():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    return net.layers['19/Fused_Add_'].out_data[0]


def test_name(recwarn):
    warnings.simplefilter("always")
    assert layer_out_data().name == "19/Fused_Add_", "Incorrect name for layer '19/Fused_Add_'"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_precision(recwarn):
    warnings.simplefilter("always")
    assert layer_out_data().precision == "FP32", "Incorrect precision for layer '19/Fused_Add_'"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_precision_setter(recwarn):
    warnings.simplefilter("always")
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.layers['19/Fused_Add_'].out_data[0].precision = "I8"
    assert net.layers['19/Fused_Add_'].out_data[0].precision == "I8", "Incorrect precision for layer '19/Fused_Add_'"
    assert len(recwarn) == 2
    assert recwarn.pop(DeprecationWarning)


def test_incorrect_precision_setter(recwarn):
    warnings.simplefilter("always")
    with pytest.raises(ValueError) as e:
        layer_out_data().precision = "123"
    assert "Unsupported precision 123! List of supported precisions:" in str(e.value)
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_layout(recwarn):
    warnings.simplefilter("always")
    assert layer_out_data().layout == "NCHW", "Incorrect layout for layer '19/Fused_Add_'"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)


def test_initialized(recwarn):
    warnings.simplefilter("always")
    assert layer_out_data().initialized, "Incorrect value for initialized property for layer '19/Fused_Add_'"
    assert len(recwarn) == 1
    assert recwarn.pop(DeprecationWarning)
