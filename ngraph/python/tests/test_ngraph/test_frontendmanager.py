# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager, FrontEnd, FrontEndCapabilities, InputModel, Place, PartialShape
from ngraph import IFrontEnd, IInputModel, IPlace
import pytest

class MockPlace(IPlace):
    def __init__(self, _isInput, _isOutput, _names):
        self.m_isInput = _isInput
        self.m_isOutput = _isOutput
        self.m_names = _names
        self.m_eqPlace = None
        self.m_isEqualCnt = 0
        self.m_isInputCnt = 0
        self.m_isOutputCnt = 0
        self.m_getNamesCnt = 0
        super(MockPlace, self).__init__()

    def set_equal(self, place):
        self.eqPlace = place

    def is_input(self):
        # print("MockPlace: Called is_input {}".format(self.m_isInput))
        self.m_isInputCnt += 1
        return self.m_isInput

    def is_output(self):
        # print("MockPlace: Called is_output {}".format(self.m_isOutput))
        self.m_isOutputCnt += 1
        return self.m_isOutput

    def get_names(self):
        # print("MockPlace: Called getNames {}".format(self.m_names))
        self.m_getNamesCnt += 1
        return self.m_names

    def is_equal(self, other):
        self.m_isEqualCnt += 1
        return self.eqPlace == other


class MockInputModel(IInputModel):
    def __init__(self, inputPlace, outputPlace):
        self.m_inputPlace = inputPlace
        self.m_outputPlace = outputPlace
        self.m_getInputsCnt = 0
        self.m_getOutputsCnt = 0
        self.m_overrideInputsCnt = 0
        self.m_overrideOutputsCnt = 0
        self.m_getPlaceByTensorNameCnt = 0
        self.m_extractSubgraphCnt = 0
        self.m_setPartialShapeCnt = 0
        self.m_partialShape = []
        super(MockInputModel, self).__init__()

    def get_inputs(self):
        # print("Mock: Called get_inputs")
        self.m_getInputsCnt += 1
        return [self.m_inputPlace]

    def get_outputs(self):
        # print("Mock: Called get_outputs")
        self.m_getOutputsCnt += 1
        return [self.m_outputPlace]

    def override_all_inputs(self, inputs):
        # print("Mock: Called override_all_inputs")
        self.m_overrideInputsCnt += 1
        return

    def override_all_outputs(self, outputs):
        # print("Mock: Called override_all_outputs")
        self.m_overrideOutputsCnt += 1
        return

    def extract_subgraph(self, inputs, outputs):
        # print("Mock: Called extract_subgraph")
        self.m_extractSubgraphCnt += 1
        return

    def get_place_by_tensor_name(self, name):
        # print("Mock: Called getPlaceByTensorName {}".format(name))
        self.m_getPlaceByTensorNameCnt += 1
        if 'Input' in name:
            return self.m_inputPlace
        elif 'Output' in name:
            return self.m_outputPlace
        return None

    def set_partial_shape(self, place, shape):
        # print("Mock: Called set_partial_shape")
        self.m_setPartialShapeCnt += 1
        self.m_partialShape = shape


class MockFrontEnd(IFrontEnd):
    def __init__(self, mdl):
        self.mockModel = mdl
        self.loadCnt = 0
        self.convertCnt = 0
        super(MockFrontEnd, self).__init__()

    def load_from_file(self, path):
        self.loadCnt += 1
        # print("Mock: Called load_from_file: {}".format(path))
        return self.mockModel

    def convert(self, model):
        self.convertCnt += 1
        # print("Mock: Called convert {}".format(model))
        return None


def setup(feName="mock"):
    # -------------- Setup mock objects ----------------
    mockInputPlace = MockPlace(True, False, ['nameInput1', 'nameInput2'])
    mockInputPlace.set_equal(mockInputPlace)
    mockOutputPlace = MockPlace(False, True, ['nameOutput1', 'nameOutput2', 'nameOutput3'])
    mockOutputPlace.set_equal(mockOutputPlace)
    mockModel = MockInputModel(mockInputPlace, mockOutputPlace)
    mockFe = MockFrontEnd(mockModel)

    def createMock(fec):
        # print("Create Mock Called, caps = {}".format(fec))
        assert fec == FrontEndCapabilities.WILDCARDS
        return mockFe

    fem = FrontEndManager()
    fem.register_front_end(feName, createMock)
    frontEnds = fem.available_front_ends()
    # print("Available frontends: {}".format(frontEnds))
    assert feName in frontEnds
    return fem, mockFe, mockModel, mockInputPlace, mockOutputPlace


def test_frontendmanager():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()

    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    assert fe is not None
    model = fe.load_from_file("abc.bin")
    assert mockFe.loadCnt == 1


def test_get_inputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    inputs = model.get_inputs()
    assert mockModel.m_getInputsCnt == 1
    assert len(inputs) == 1
    assert inputs[0].is_input()
    assert mockInputPlace.m_isInputCnt == 1
    assert not inputs[0].is_output()
    assert mockInputPlace.m_isOutputCnt == 1
    assert len(inputs[0].get_names()) == 2
    assert mockInputPlace.m_getNamesCnt == 1


def test_get_outputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    outputs = model.get_outputs()
    assert mockModel.m_getOutputsCnt == 1
    assert len(outputs) == 1
    assert outputs[0].is_output()
    assert mockOutputPlace.m_isOutputCnt == 1
    assert not outputs[0].is_input()
    assert mockOutputPlace.m_isInputCnt == 1
    assert len(outputs[0].get_names()) == 3
    assert mockOutputPlace.m_getNamesCnt == 1


def test_is_equal():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    outputs = model.get_outputs()
    inputs = model.get_inputs()
    assert inputs[0].is_equal(inputs[0])
    assert mockInputPlace.m_isEqualCnt == 1
    assert outputs[0].is_equal(outputs[0])
    assert mockOutputPlace.m_isEqualCnt == 1
    assert not inputs[0].is_equal(outputs[0])
    assert mockInputPlace.m_isEqualCnt == 2
    assert not outputs[0].is_equal(inputs[0])
    assert mockOutputPlace.m_isEqualCnt == 2


def test_override_inputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    inputs = model.get_inputs()
    model.override_all_inputs(inputs)
    assert mockModel.m_overrideInputsCnt == 1


def test_override_outputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    outputs = model.get_outputs()
    model.override_all_outputs(outputs)
    assert mockModel.m_overrideOutputsCnt == 1


def test_extract_subgraph():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    outputs = model.get_outputs()
    inputs = model.get_inputs()
    model.extract_subgraph(inputs, outputs)
    assert mockModel.m_extractSubgraphCnt == 1


def test_set_partial_shape():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    inputs = model.get_inputs()
    model.set_partial_shape(inputs[0], PartialShape([1, 2, 3, 4]))
    assert mockModel.m_setPartialShapeCnt == 1
    assert mockModel.m_partialShape == PartialShape([1, 2, 3, 4])


def test_get_place_by_tensor_name():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    place = model.get_place_by_tensor_name('nameInput1')
    assert mockModel.m_getPlaceByTensorNameCnt == 1
    assert place.is_input()

    place = model.get_place_by_tensor_name('nameOutput2')
    assert mockModel.m_getPlaceByTensorNameCnt == 2
    assert place.is_output()


def test_convert():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.load_by_framework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.load_from_file("abc.bin")

    fe.convert(model)
    assert mockFe.convertCnt == 1


def test_frontend_caps():
    capabilities = FrontEndCapabilities.WILDCARDS
    assert str(capabilities)
