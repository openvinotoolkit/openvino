# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pickle

from ngraph import PartialShape
from ngraph.frontend import FrontEndManager, InitializationFailure
from ngraph.utils.types import get_element_type

import numpy as np

import pytest

mock_available = True
try:
    from pybind_mock_frontend import get_fe_stat, get_mdl_stat, get_place_stat
except Exception:
    print("No mock frontend available")
    mock_available = False

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()

mock_needed = pytest.mark.skipif(not mock_available,
                                 reason="mock fe is not available")


# ---------- FrontEnd tests ---------------
def test_pickle():
    pickle.dumps(fem)


def test_load_by_unknown_framework():
    frontEnds = fem.get_available_front_ends()
    assert not ("UnknownFramework" in frontEnds)
    try:
        fem.load_by_framework("UnknownFramework")
    except InitializationFailure as exc:
        print(exc)
    else:
        raise AssertionError("Unexpected exception.")


@mock_needed
def test_load():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load("abc.bin")
    assert model is not None
    stat = get_fe_stat(fe)
    assert "abc.bin" in stat.load_paths


@mock_needed
def test_convert_model():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load(path="")
    func = fe.convert(model=model)
    assert func is not None
    stat = get_fe_stat(fe)
    assert stat.convert_model == 1


@mock_needed
def test_convert_partially():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load(path="")
    func = fe.convert_partially(model=model)
    stat = get_fe_stat(fe)
    assert stat.convert_partially == 1
    fe.convert(function=func)
    stat = get_fe_stat(fe)
    assert stat.convert == 1


@mock_needed
def test_decode_and_normalize():
    fe = fem.load_by_framework(framework="mock_py")
    assert fe is not None
    model = fe.load(path="")
    func = fe.decode(model=model)
    stat = get_fe_stat(fe)
    assert stat.decode == 1
    fe.normalize(function=func)
    stat = get_fe_stat(fe)
    assert stat.normalize == 1
    assert stat.decode == 1


# --------InputModel tests-----------------
@mock_needed
def init_model():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load(path="")
    return model


@mock_needed
def test_model_get_inputs():
    model = init_model()
    for i in range(1, 10):
        model.get_inputs()
        stat = get_mdl_stat(model)
        assert stat.get_inputs == i


@mock_needed
def test_model_get_outputs():
    model = init_model()
    for i in range(1, 10):
        model.get_outputs()
        stat = get_mdl_stat(model)
        assert stat.get_outputs == i


@mock_needed
def test_model_get_place_by_tensor_name():
    model = init_model()
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_tensor_name(tensorName=name)
        stat = get_mdl_stat(model)
        assert stat.get_place_by_tensor_name == i
        assert stat.lastArgString == name


@mock_needed
def test_model_get_place_by_operation_name():
    model = init_model()
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_name(operationName=name)
        stat = get_mdl_stat(model)
        assert stat.get_place_by_operation_name == i
        assert stat.lastArgString == name


@mock_needed
def test_model_get_place_by_operation_name_and_input_port():
    model = init_model()
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_name_and_input_port(operationName=name, inputPortIndex=i * 2)
        stat = get_mdl_stat(model)
        assert stat.get_place_by_operation_and_input_port == i
        assert stat.lastArgString == name
        assert stat.lastArgInt == i * 2


@mock_needed
def test_model_get_place_by_operation_name_and_output_port():
    model = init_model()
    for i in range(1, 10):
        name = str(i)
        model.get_place_by_operation_name_and_output_port(operationName=name, outputPortIndex=i * 2)
        stat = get_mdl_stat(model)
        assert stat.get_place_by_operation_and_output_port == i
        assert stat.lastArgString == name
        assert stat.lastArgInt == i * 2


@mock_needed
def test_model_set_name_for_tensor():
    model = init_model()
    place = model.get_place_by_tensor_name(tensorName="")
    model.set_name_for_tensor(tensor=place, newName="1234")
    stat = get_mdl_stat(model)
    assert stat.set_name_for_tensor == 1
    assert stat.lastArgString == "1234"
    assert stat.lastArgPlace == place


@mock_needed
def test_model_add_name_for_tensor():
    model = init_model()
    place = model.get_place_by_tensor_name(tensorName="")
    model.add_name_for_tensor(tensor=place, newName="1234")
    stat = get_mdl_stat(model)
    assert stat.add_name_for_tensor == 1
    assert stat.lastArgString == "1234"
    assert stat.lastArgPlace == place


@mock_needed
def test_model_set_name_for_operation():
    model = init_model()
    place = model.get_place_by_operation_name(operationName="")
    model.set_name_for_operation(operation=place, newName="1111")
    stat = get_mdl_stat(model)
    assert stat.set_name_for_operation == 1
    assert stat.lastArgString == "1111"
    assert stat.lastArgPlace == place


@mock_needed
def test_model_free_name_for_tensor():
    model = init_model()
    model.free_name_for_tensor(name="2222")
    stat = get_mdl_stat(model)
    assert stat.free_name_for_tensor == 1
    assert stat.lastArgString == "2222"


@mock_needed
def test_model_free_name_for_operation():
    model = init_model()
    model.free_name_for_operation(name="3333")
    stat = get_mdl_stat(model)
    assert stat.free_name_for_operation == 1
    assert stat.lastArgString == "3333"


@mock_needed
def test_model_set_name_for_dimension():
    model = init_model()
    place = model.get_place_by_operation_name(operationName="")
    model.set_name_for_dimension(place=place, dimIndex=123, dimName="4444")
    stat = get_mdl_stat(model)
    assert stat.set_name_for_dimension == 1
    assert stat.lastArgString == "4444"
    assert stat.lastArgInt == 123
    assert stat.lastArgPlace == place


@mock_needed
def test_model_cut_and_add_new_input():
    model = init_model()
    place = model.get_place_by_operation_name("")
    model.cut_and_add_new_input(place=place, newName="5555")
    stat = get_mdl_stat(model)
    assert stat.cut_and_add_new_input == 1
    assert stat.lastArgString == "5555"
    assert stat.lastArgPlace == place
    model.cut_and_add_new_input(place=place)
    stat = get_mdl_stat(model)
    assert stat.cut_and_add_new_input == 2
    assert stat.lastArgString == ""
    assert stat.lastArgPlace == place


@mock_needed
def test_model_cut_and_add_new_output():
    model = init_model()
    place = model.get_place_by_operation_name("")
    model.cut_and_add_new_output(place=place, newName="5555")
    stat = get_mdl_stat(model)
    assert stat.cut_and_add_new_output == 1
    assert stat.lastArgString == "5555"
    assert stat.lastArgPlace == place
    model.cut_and_add_new_output(place=place)
    stat = get_mdl_stat(model)
    assert stat.cut_and_add_new_output == 2
    assert stat.lastArgString == ""
    assert stat.lastArgPlace == place


@mock_needed
def test_model_add_output():
    model = init_model()
    place = model.get_place_by_operation_name("")
    place2 = model.add_output(place=place)
    assert place2 is not None
    stat = get_mdl_stat(model)
    assert stat.add_output == 1
    assert stat.lastArgPlace == place


@mock_needed
def test_model_remove_output():
    model = init_model()
    place = model.get_place_by_operation_name("")
    model.remove_output(place=place)
    stat = get_mdl_stat(model)
    assert stat.remove_output == 1
    assert stat.lastArgPlace == place


@mock_needed
def test_model_set_partial_shape():
    model = init_model()
    place = model.get_place_by_tensor_name(tensorName="")
    test_shape = PartialShape([1, 2, 3, 4])
    model.set_partial_shape(place=place, shape=test_shape)
    stat = get_mdl_stat(model)
    assert stat.set_partial_shape == 1
    assert stat.lastArgPlace == place
    assert stat.lastArgPartialShape == test_shape


@mock_needed
def test_model_get_partial_shape():
    model = init_model()
    place = model.get_place_by_tensor_name(tensorName="")
    shape = model.get_partial_shape(place=place)
    assert shape is not None
    stat = get_mdl_stat(model)
    assert stat.get_partial_shape == 1
    assert stat.lastArgPlace == place


@mock_needed
def test_model_override_all_inputs():
    model = init_model()
    place1 = model.get_place_by_tensor_name(tensorName="p1")
    place2 = model.get_place_by_tensor_name(tensorName="p2")
    model.override_all_inputs(inputs=[place1, place2])
    stat = get_mdl_stat(model)
    assert stat.override_all_inputs == 1
    assert len(stat.lastArgInputPlaces) == 2
    assert stat.lastArgInputPlaces[0] == place1
    assert stat.lastArgInputPlaces[1] == place2


@mock_needed
def test_model_override_all_outputs():
    model = init_model()
    place1 = model.get_place_by_tensor_name(tensorName="p1")
    place2 = model.get_place_by_tensor_name(tensorName="p2")
    model.override_all_outputs(outputs=[place1, place2])
    stat = get_mdl_stat(model)
    assert stat.override_all_outputs == 1
    assert len(stat.lastArgOutputPlaces) == 2
    assert stat.lastArgOutputPlaces[0] == place1
    assert stat.lastArgOutputPlaces[1] == place2


@mock_needed
def test_model_extract_subgraph():
    model = init_model()
    place1 = model.get_place_by_tensor_name(tensorName="p1")
    place2 = model.get_place_by_tensor_name(tensorName="p2")
    place3 = model.get_place_by_tensor_name(tensorName="p3")
    place4 = model.get_place_by_tensor_name(tensorName="p4")
    model.extract_subgraph(inputs=[place1, place2], outputs=[place3, place4])
    stat = get_mdl_stat(model)
    assert stat.extract_subgraph == 1
    assert len(stat.lastArgInputPlaces) == 2
    assert stat.lastArgInputPlaces[0] == place1
    assert stat.lastArgInputPlaces[1] == place2
    assert len(stat.lastArgOutputPlaces) == 2
    assert stat.lastArgOutputPlaces[0] == place3
    assert stat.lastArgOutputPlaces[1] == place4


@mock_needed
def test_model_set_element_type():
    model = init_model()
    place = model.get_place_by_tensor_name(tensorName="")
    model.set_element_type(place=place, type=get_element_type(np.int32))
    stat = get_mdl_stat(model)
    assert stat.set_element_type == 1
    assert stat.lastArgPlace == place
    assert stat.lastArgElementType == get_element_type(np.int32)


# ----------- Place test ------------
@mock_needed
def init_place():
    fe = fem.load_by_framework(framework="mock_py")
    model = fe.load(path="")
    place = model.get_place_by_tensor_name(tensorName="")
    return model, place


@mock_needed
def test_place_is_input():
    _, place = init_place()
    assert place.is_input() is not None
    stat = get_place_stat(place)
    assert stat.is_input == 1


@mock_needed
def test_place_is_output():
    _, place = init_place()
    assert place.is_output() is not None
    stat = get_place_stat(place)
    assert stat.is_output == 1


@mock_needed
def test_place_get_names():
    _, place = init_place()
    assert place.get_names() is not None
    stat = get_place_stat(place)
    assert stat.get_names == 1


@mock_needed
def test_place_is_equal():
    model, place = init_place()
    place2 = model.get_place_by_tensor_name("2")
    assert place.is_equal(other=place2) is not None
    stat = get_place_stat(place)
    assert stat.is_equal == 1
    assert stat.lastArgPlace == place2


@mock_needed
def test_place_is_equal_data():
    model, place = init_place()
    place2 = model.get_place_by_tensor_name("2")
    assert place.is_equal_data(other=place2) is not None
    stat = get_place_stat(place)
    assert stat.is_equal_data == 1
    assert stat.lastArgPlace == place2


@mock_needed
def test_place_get_consuming_operations():
    _, place = init_place()
    assert place.get_consuming_operations(outputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_consuming_operations == 1
    assert stat.lastArgInt == 22
    assert place.get_consuming_operations() is not None
    stat = get_place_stat(place)
    assert stat.get_consuming_operations == 2
    assert stat.lastArgInt == -1


@mock_needed
def test_place_get_target_tensor():
    _, place = init_place()
    assert place.get_target_tensor(outputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_target_tensor == 1
    assert stat.lastArgInt == 22
    assert place.get_target_tensor() is not None
    stat = get_place_stat(place)
    assert stat.get_target_tensor == 2
    assert stat.lastArgInt == -1


@mock_needed
def test_place_get_producing_operation():
    _, place = init_place()
    assert place.get_producing_operation(inputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_producing_operation == 1
    assert stat.lastArgInt == 22
    assert place.get_producing_operation() is not None
    stat = get_place_stat(place)
    assert stat.get_producing_operation == 2
    assert stat.lastArgInt == -1


@mock_needed
def test_place_get_producing_port():
    _, place = init_place()
    assert place.get_producing_port() is not None
    stat = get_place_stat(place)
    assert stat.get_producing_port == 1


@mock_needed
def test_place_get_input_port():
    _, place = init_place()
    assert place.get_input_port() is not None
    stat = get_place_stat(place)
    assert stat.get_input_port == 1
    assert stat.lastArgInt == -1
    assert place.get_input_port(inputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_input_port == 2
    assert stat.lastArgInt == 22


@mock_needed
def test_place_get_input_port2():
    _, place = init_place()
    assert place.get_input_port(inputName="abc") is not None
    stat = get_place_stat(place)
    assert stat.get_input_port == 1
    assert stat.lastArgInt == -1
    assert stat.lastArgString == "abc"
    assert place.get_input_port(inputName="abcd", inputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_input_port == 2
    assert stat.lastArgInt == 22
    assert stat.lastArgString == "abcd"


@mock_needed
def test_place_get_output_port():
    _, place = init_place()
    assert place.get_output_port() is not None
    stat = get_place_stat(place)
    assert stat.get_output_port == 1
    assert stat.lastArgInt == -1
    assert place.get_output_port(outputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_output_port == 2
    assert stat.lastArgInt == 22


@mock_needed
def test_place_get_output_port2():
    _, place = init_place()
    assert place.get_output_port(outputName="abc") is not None
    stat = get_place_stat(place)
    assert stat.get_output_port == 1
    assert stat.lastArgInt == -1
    assert stat.lastArgString == "abc"
    assert place.get_output_port(outputName="abcd", outputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_output_port == 2
    assert stat.lastArgInt == 22
    assert stat.lastArgString == "abcd"


@mock_needed
def test_place_get_consuming_ports():
    _, place = init_place()
    assert place.get_consuming_ports() is not None
    stat = get_place_stat(place)
    assert stat.get_consuming_ports == 1


@mock_needed
def test_place_get_source_tensor():
    _, place = init_place()
    assert place.get_source_tensor() is not None
    stat = get_place_stat(place)
    assert stat.get_source_tensor == 1
    assert stat.lastArgInt == -1
    assert place.get_source_tensor(inputPortIndex=22) is not None
    stat = get_place_stat(place)
    assert stat.get_source_tensor == 2
    assert stat.lastArgInt == 22
