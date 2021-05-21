# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ngraph import FrontEndManager, FrontEndCapabilities, PartialShape
from ngraph.utils.types import get_element_type

from pybind_mock_frontend import get_fe_stat, reset_fe_stat, FeStat
from pybind_mock_frontend import get_mdl_stat, reset_mdl_stat, ModelStat

# Shall be initialized and destroyed after all tests finished
# This is required as destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()


# ---------- FrontEnd tests ---------------
def test_load_by_framework_caps():
    frontEnds = fem.get_available_front_ends()
    assert frontEnds is not None
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
    for i in range(len(caps) - 1):
        for j in range(i+1, len(caps)):
            assert caps[i] != caps[j]


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


# --------InputModel tests-----------------
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
        assert stat.lastArgString == name


def test_getPlaceByOperationName():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_name(name)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameCount == i
        assert stat.lastArgString == name


def test_getPlaceByOperationAndInputPort():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_and_input_port(name, i * 2)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameAndInputPortCount == i
        assert stat.lastArgString == name
        assert stat.lastArgInt == i * 2


def test_getPlaceByOperationAndOutputPort():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_and_output_port(name, i * 2)
        stat = get_mdl_stat(model)
        assert stat.getPlaceByOperationNameAndOutputPortCount == i
        assert stat.lastArgString == name
        assert stat.lastArgInt == i * 2


def test_setNameForTensor():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_tensor_name("")
    model.set_name_for_tensor(place, "1234")
    stat = get_mdl_stat(model)
    assert stat.setNameForTensorCount == 1
    assert stat.lastArgString == "1234"
    assert stat.lastArgPlace == place


def test_addNameForTensor():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_tensor_name("")
    model.add_name_for_tensor(place, "1234")
    stat = get_mdl_stat(model)
    assert stat.addNameForTensorCount == 1
    assert stat.lastArgString == "1234"
    assert stat.lastArgPlace == place


def test_setNameForOperation():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    model.set_name_for_operation(place, "1111")
    stat = get_mdl_stat(model)
    assert stat.setNameForOperationCount == 1
    assert stat.lastArgString == "1111"
    assert stat.lastArgPlace == place


def test_freeNameForTensor():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    model.free_name_for_tensor("2222")
    stat = get_mdl_stat(model)
    assert stat.freeNameForTensorCount == 1
    assert stat.lastArgString == "2222"


def test_freeNameForOperation():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    model.free_name_for_operation("3333")
    stat = get_mdl_stat(model)
    assert stat.freeNameForOperationCount == 1
    assert stat.lastArgString == "3333"


def test_setNameForDimension():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    model.set_name_for_dimension(place, 123, "4444")
    stat = get_mdl_stat(model)
    assert stat.setNameForDimensionCount == 1
    assert stat.lastArgString == "4444"
    assert stat.lastArgInt == 123
    assert stat.lastArgPlace == place



def test_cutAndAddNewInput():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    model.cut_and_add_new_input(place, "5555")
    stat = get_mdl_stat(model)
    assert stat.cutAndAddNewInputCount == 1
    assert stat.lastArgString == "5555"
    assert stat.lastArgPlace == place


def test_cutAndAddNewOutput():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    model.cut_and_add_new_output(place, "5555")
    stat = get_mdl_stat(model)
    assert stat.cutAndAddNewOutputCount == 1
    assert stat.lastArgString == "5555"
    assert stat.lastArgPlace == place


def test_addOutput():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    place2 = model.add_output(place)
    assert place2 is not None
    stat = get_mdl_stat(model)
    assert stat.addOutputCount == 1
    assert stat.lastArgPlace == place


def test_removeOutput():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_operation_name("")
    model.remove_output(place)
    stat = get_mdl_stat(model)
    assert stat.removeOutputCount == 1
    assert stat.lastArgPlace == place


def test_setPartialShape():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_tensor_name("")
    test_shape = PartialShape([1, 2, 3, 4])
    model.set_partial_shape(place, test_shape)
    stat = get_mdl_stat(model)
    assert stat.setPartialShapeCount == 1
    assert stat.lastArgPlace == place
    assert stat.lastArgPartialShape == test_shape


def test_getPartialShape():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_tensor_name("")
    shape = model.get_partial_shape(place)
    assert shape is not None
    stat = get_mdl_stat(model)
    assert stat.getPartialShapeCount == 1
    assert stat.lastArgPlace == place


def test_overrideAllInputs():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place1 = model.get_place_by_tensor_name("p1")
    place2 = model.get_place_by_tensor_name("p2")
    model.override_all_inputs([place1, place2])
    stat = get_mdl_stat(model)
    assert stat.overrideAllInputsCount == 1
    assert len(stat.lastArgInputPlaces) == 2
    assert stat.lastArgInputPlaces[0] == place1
    assert stat.lastArgInputPlaces[1] == place2


def test_overrideAllOutputs():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place1 = model.get_place_by_tensor_name("p1")
    place2 = model.get_place_by_tensor_name("p2")
    model.override_all_outputs([place1, place2])
    stat = get_mdl_stat(model)
    assert stat.overrideAllOutputsCount == 1
    assert len(stat.lastArgOutputPlaces) == 2
    assert stat.lastArgOutputPlaces[0] == place1
    assert stat.lastArgOutputPlaces[1] == place2


def test_extractSubgraph():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place1 = model.get_place_by_tensor_name("p1")
    place2 = model.get_place_by_tensor_name("p2")
    place3 = model.get_place_by_tensor_name("p3")
    place4 = model.get_place_by_tensor_name("p4")
    model.extract_subgraph([place1, place2], [place3, place4])
    stat = get_mdl_stat(model)
    assert stat.extractSubgraphCount == 1
    assert len(stat.lastArgInputPlaces) == 2
    assert stat.lastArgInputPlaces[0] == place1
    assert stat.lastArgInputPlaces[1] == place2
    assert len(stat.lastArgOutputPlaces) == 2
    assert stat.lastArgOutputPlaces[0] == place3
    assert stat.lastArgOutputPlaces[1] == place4


def test_setElementType():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load_from_file("")
    place = model.get_place_by_tensor_name("")
    model.set_element_type(place, get_element_type(np.int32))
    stat = get_mdl_stat(model)
    assert stat.setElementTypeCount == 1
    assert stat.lastArgPlace == place
    assert stat.lastArgElementType == get_element_type(np.int32)


# if __name__ == '__main__':
#     test_frontendmanager()