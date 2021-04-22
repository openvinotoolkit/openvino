# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager, FrontEnd, FrontEndCapabilities, InputModel, Place, PartialShape
import pytest

class MockPlace(Place):
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
        # print("MockPlace: Called isInput {}".format(self.m_isInput))
        self.m_isInputCnt += 1
        return self.m_isInput

    def is_output(self):
        # print("MockPlace: Called isOutput {}".format(self.m_isOutput))
        self.m_isOutputCnt += 1
        return self.m_isOutput

    def get_names(self):
        # print("MockPlace: Called getNames {}".format(self.m_names))
        self.m_getNamesCnt += 1
        return self.m_names

    def is_equal(self, other):
        self.m_isEqualCnt += 1
        return self.eqPlace == other


class MockInputModel(InputModel):
    def __init__(self, inputPlace, outputPlace):
        self.m_inputPlace = inputPlace
        self.m_outputPlace = outputPlace
        self.m_getInputsCnt = 0
        self.m_getOutputsCnt = 0
        self.m_overrideInputsCnt = 0
        self.m_overrideOutputsCnt = 0
        self.m_getPlaceByTensorNameCnt = 0
        self.m_extractSubGraph = 0
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
        self.m_extractSubGraph += 1
        return

    def get_place_by_tensor_name(self, name):
        # print("Mock: Called get_place_by_tensor_name {}".format(name))
        self.m_getPlaceByTensorNameCnt += 1
        if 'Input' in name:
            return self.m_inputPlace
        elif 'Output' in name:
            return self.m_outputPlace
        return None

    def set_partial_shape(self, place, shape):
        # print("Mock: Called set_partial_shape {}".format(name))
        self.m_setPartialShapeCnt += 1
        self.m_partialShape = shape


class MockFrontEnd(FrontEnd):
    def __init__(self, mdl):
        self.mockModel = mdl
        self.loadCnt = 0
        self.convertCnt = 0
        super(MockFrontEnd, self).__init__()

    def load_from_file(self, path):
        self.loadCnt += 1
        # print("Mock: Called doLoadFromFile: {}".format(path))
        return self.mockModel

    def do_convert(self, model):
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
    fem.registerFrontEnd(feName, createMock)
    frontEnds = fem.availableFrontEnds()
    # print("Available frontends: {}".format(frontEnds))
    assert feName in frontEnds
    return fem, mockFe, mockModel, mockInputPlace, mockOutputPlace


def test_frontendmanager():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()

    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    assert fe is not None
    model = fe.loadFromFile("abc.bin")
    assert mockFe.loadCnt == 1


def test_get_inputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    inputs = model.getInputs()
    assert mockModel.m_getInputsCnt == 1
    assert len(inputs) == 1
    assert inputs[0].isInput()
    assert mockInputPlace.m_isInputCnt == 1
    assert not inputs[0].isOutput()
    assert mockInputPlace.m_isOutputCnt == 1
    assert len(inputs[0].getNames()) == 2
    assert mockInputPlace.m_getNamesCnt == 1


def test_get_outputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    outputs = model.getOutputs()
    assert mockModel.m_getOutputsCnt == 1
    assert len(outputs) == 1
    assert outputs[0].isOutput()
    assert mockOutputPlace.m_isOutputCnt == 1
    assert not outputs[0].isInput()
    assert mockOutputPlace.m_isInputCnt == 1
    assert len(outputs[0].getNames()) == 3
    assert mockOutputPlace.m_getNamesCnt == 1


def test_is_equal():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    outputs = model.getOutputs()
    inputs = model.getInputs()
    assert inputs[0].isEqual(inputs[0])
    assert mockInputPlace.m_isEqualCnt == 1
    assert outputs[0].isEqual(outputs[0])
    assert mockOutputPlace.m_isEqualCnt == 1
    assert not inputs[0].isEqual(outputs[0])
    assert mockInputPlace.m_isEqualCnt == 2
    assert not outputs[0].isEqual(inputs[0])
    assert mockOutputPlace.m_isEqualCnt == 2


def test_override_inputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    inputs = model.getInputs()
    model.overrideAllInputs(inputs)
    assert mockModel.m_overrideInputsCnt == 1


def test_override_outputs():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    outputs = model.getOutputs()
    model.overrideAllOutputs(outputs)
    assert mockModel.m_overrideOutputsCnt == 1


def test_extract_subgraph():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    outputs = model.getOutputs()
    inputs = model.getInputs()
    model.extractSubgraph(inputs, outputs)
    assert mockModel.m_extractSubGraph == 1


def test_set_partial_shape():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    inputs = model.getInputs()
    model.setPartialShape(inputs[0], PartialShape([1, 2, 3, 4]))
    assert mockModel.m_setPartialShapeCnt == 1
    assert mockModel.m_partialShape == PartialShape([1, 2, 3, 4])


def test_get_place_by_tensor_name():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    place = model.getPlaceByTensorName('nameInput1')
    assert mockModel.m_getPlaceByTensorNameCnt == 1
    assert place.isInput()

    place = model.getPlaceByTensorName('nameOutput2')
    assert mockModel.m_getPlaceByTensorNameCnt == 2
    assert place.isOutput()


def test_convert():
    fem, mockFe, mockModel, mockInputPlace, mockOutputPlace = setup()
    fe = fem.loadByFramework(framework="mock", capabilities=FrontEndCapabilities.WILDCARDS)
    model = fe.loadFromFile("abc.bin")

    fe.convert(model)
    assert mockFe.convertCnt == 1


def test_error_cases():
    fem1, mockFe1, mockModel1, mockInputPlace1, mockOutputPlace1 = setup("mock1")
    fem2, mockFe2, mockModel2, mockInputPlace2, mockOutputPlace2 = setup("mock2")
    fe1 = fem1.loadByFramework(framework="mock1", capabilities=FrontEndCapabilities.WILDCARDS)
    fe2 = fem2.loadByFramework(framework="mock2", capabilities=FrontEndCapabilities.WILDCARDS)
    model1 = fe1.loadFromFile("abc.bin")
    model2 = fe2.loadFromFile("abc.bin")
    inputs1 = model1.getInputs()
    inputs2 = model2.getInputs()
    outputs1 = model1.getOutputs()
    outputs2 = model2.getOutputs()

    with pytest.raises(RuntimeError) as excInfo:
        fe1.convert(model2)
    assert 'convert' in str(excInfo.value)

    with pytest.raises(RuntimeError) as excInfo:
        model1.overrideAllInputs(inputs2)
    assert 'Place' in str(excInfo.value)

    with pytest.raises(RuntimeError) as excInfo:
        model1.overrideAllOutputs(inputs2)
    assert 'Place' in str(excInfo.value)

    with pytest.raises(RuntimeError) as excInfo:
        model1.extractSubgraph(inputs1, outputs2)
    assert 'Place' in str(excInfo.value)

    with pytest.raises(RuntimeError) as excInfo:
        model1.extractSubgraph(inputs2, outputs1)
    assert 'Place' in str(excInfo.value)

    with pytest.raises(RuntimeError) as excInfo:
        model1.setPartialShape(inputs2[0], PartialShape([1, 2, 3, 4]))
    assert 'Place' in str(excInfo.value)


def test_frontend_caps():
    capabilities = FrontEndCapabilities.WILDCARDS
    assert str(capabilities)
