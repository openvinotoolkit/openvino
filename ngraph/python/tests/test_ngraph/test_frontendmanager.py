# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager, FrontEndCapabilities

from pybind_mock_frontend import get_fe_stat, reset_fe_stat, FeStat
from pybind_mock_frontend import get_mdl_stat, reset_mdl_stat, ModelStat

# Shall be initialized and destroyed after every test finished
# This is required as destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()

def test_load_by_framework_caps():
    frontEnds = fem.get_available_front_ends()
    assert frontEnds is not None
    print("FrontEnds: {}".format(frontEnds))
    assert 'mock_py' in frontEnds
    caps = [FrontEndCapabilities.DEFAULT,
            FrontEndCapabilities.CUT,
            FrontEndCapabilities.NAMES,
            FrontEndCapabilities.WILDCARDS,
            FrontEndCapabilities.CUT | FrontEndCapabilities.NAMES | FrontEndCapabilities.WILDCARDS]
    for cap in caps:
        fe = fem.load_by_framework(framework="mock_py", capabilities=cap)
        stat = get_fe_stat(fe)
        assert cap == stat.load_flags


def test_load_from_file():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("abc.bin")
    stat = get_fe_stat(fe)
    assert 'abc.bin' in stat.loaded_paths


def test_convert_model():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("")
    fe.convert(model)
    stat = get_fe_stat(fe)
    assert stat.convertModelCount == 1


def test_convert_partially():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load_from_file("")
    func = fe.convert_partially(model)
    stat = get_fe_stat(fe)
    assert stat.convertPartCount == 1
    fe.convert(func)
    stat = get_fe_stat(fe)
    assert stat.convertFuncCount == 1


def test_decode_and_normalize():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    stat = get_fe_stat(fe)
    assert stat.normalizeCount == 0
    assert stat.decodeCount == 0
    model = fe.load_from_file("")
    func = fe.decode(model)
    stat = get_fe_stat(fe)
    assert stat.normalizeCount == 0
    assert stat.decodeCount == 1
    fe.normalize(func)
    stat = get_fe_stat(fe)
    assert stat.normalizeCount == 1
    assert stat.decodeCount == 1


#--------InputModel tests-----------------

def test_getInputs():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        model.get_inputs()
        stat = get_mdl_stat(model)
        assert stat.getInputsCount == i


def test_getOutputs():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        model.get_outputs()
        stat = get_mdl_stat(model)
        assert stat.getOutputsCount == i


def test_getPlaceByTensorName():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_tensor_name(name)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByTensorNameCount == i
        assert stat.lastPlaceName == name


def test_getPlaceByOperationName():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_name(name)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameCount == i
        assert stat.lastPlaceName == name


def test_getPlaceByOperationAndInputPort():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_and_input_port(name, i * 2)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameAndInputPortCount == i
        assert stat.lastPlaceName == name
        assert stat.lastPlacePortIndex == i * 2


def test_getPlaceByOperationAndOutputPort():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_and_output_port(name, i * 2)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameAndOutputPortCount == i
        assert stat.lastPlaceName == name
        assert stat.lastPlacePortIndex == i * 2

# if __name__ == '__main__':
#     test_frontendmanager()