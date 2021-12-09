// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <vpu/ngraph/operations/static_shape_nonzero.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace vpu {

namespace ie = InferenceEngine;

class DSRParsingTests : public GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        _testModel = CreateTestModel();
    }

    void checkShapeConnection(const Data& parent, const Data& child) {
        ASSERT_NE(child->parentDataToShapeEdge(), nullptr);
        ASSERT_EQ(child->childDataToShapeEdges().size(), 0);
        const auto& parentDataToShapeEdge = child->parentDataToShapeEdge();
        ASSERT_EQ(parentDataToShapeEdge->parent(), parent);
        ASSERT_EQ(parentDataToShapeEdge->child(), child);

        ASSERT_EQ(parent->parentDataToShapeEdge(), nullptr);

        const auto& childDataToShapeEdges = parent->childDataToShapeEdges();

        const auto& it = std::find(childDataToShapeEdges.begin(), childDataToShapeEdges.end(), parentDataToShapeEdge);
        ASSERT_NE(it, childDataToShapeEdges.end());
    }

    ie::CNNLayerPtr createDSRLayer() {
        return std::make_shared<ie::CNNLayer>(ie::LayerParams{"DSR", "DynamicShapeResolver", ie::Precision::I32});
    }

protected:
    TestModel _testModel;
    DataDesc _dataDesc = {800};
    DataDesc _correstShapeDesc = {1};
    DataDesc _incorrestShapeDesc = {2};
};

TEST_F(DSRParsingTests, DSRParserAssertsOnInputDSR) {
    _testModel.createInputs({_dataDesc, _correstShapeDesc});
    _testModel.createOutputs({_dataDesc});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_ANY_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
                                        {_testModel.getInputs()[0], _testModel.getInputs()[1]}, _testModel.getOutputs()));
}

TEST_F(DSRParsingTests, DSRParserAssertsOnIncorrectDimensions) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc});

    const auto& inputStage = _testModel.addStage({InputInfo::fromNetwork(0)},
            {OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_incorrestShapeDesc)});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_ANY_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
                                        {inputStage->output(0), inputStage->output(1)}, _testModel.getOutputs()));
}

TEST_F(DSRParsingTests, DSRParserAssertsOnIncorrectNumInputs) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc});

    const auto& inputStage = _testModel.addStage({InputInfo::fromNetwork(0)},
                                                {OutputInfo::intermediate(_dataDesc)});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_ANY_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
                                        {inputStage->output(0)}, _testModel.getOutputs()));
}

TEST_F(DSRParsingTests, DSRParserAssertsOnIncorrectNumOutputs) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc, _dataDesc});

    const auto& inputStage = _testModel.addStage({InputInfo::fromNetwork(0)},
            {OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_correstShapeDesc)});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_ANY_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
                                        {inputStage->output(0), inputStage->output(1)}, _testModel.getOutputs()));
}

TEST_F(DSRParsingTests, DSRParserDoesntAssertOnCorrectIO) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc});

    const auto& inputStage = _testModel.addStage({InputInfo::fromNetwork(0)},
                                                {OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_correstShapeDesc)});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_NO_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
                                       {inputStage->output(0), inputStage->output(1)}, _testModel.getOutputs()));
}

TEST_F(DSRParsingTests, DSRParserDoesntAssertOnTwoOutputsWithSameShapeData) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc, _dataDesc});

    const auto& inputStage = _testModel.addStage(
            {InputInfo::fromNetwork(0)},
            {OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_correstShapeDesc)});

    const auto& dsrLayer1 = createDSRLayer();
    const auto& dsrLayer2 = createDSRLayer();

    ASSERT_NO_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer1,
                                       {inputStage->output(0), inputStage->output(2)}, {_testModel.getOutputs()[0]}));
    ASSERT_NO_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer2,
                                       {inputStage->output(1), inputStage->output(2)}, {_testModel.getOutputs()[1]}));
}

TEST_F(DSRParsingTests, DSRParserPreservesConnectionsOnOutputDSR) {
    _testModel.createInputs({_dataDesc});
    _testModel.createOutputs({_dataDesc});

    const auto& model = _testModel.getBaseModel();

    const auto& inputStage = _testModel.addStage({InputInfo::fromNetwork(0)},
                                                 {OutputInfo::intermediate(_dataDesc), OutputInfo::intermediate(_correstShapeDesc)});

    model->connectDataWithShape(inputStage->output(1), inputStage->output(0));

    checkShapeConnection(inputStage->output(1), inputStage->output(0));

    const auto& outputStage = _testModel.addStage({InputInfo::fromPrevStage(0)},
                                                  {OutputInfo::intermediate(_dataDesc)});

    const auto& dsrLayer = createDSRLayer();

    ASSERT_NO_THROW(frontEnd->parseDSR(_testModel.getBaseModel(), dsrLayer,
    {outputStage->output(0), inputStage->output(1)}, _testModel.getOutputs()));

    checkShapeConnection(inputStage->output(1), inputStage->output(0));
    checkShapeConnection(inputStage->output(1), outputStage->output(0));
}

typedef DSRParsingTests DSRParsingFromNgraphTests;

TEST_F(DSRParsingFromNgraphTests, DSRParserCreatesAndConnectsTwoOutputsOnOutputDSR) {
    const auto& inPrecision = ::ngraph::element::Type(::ngraph::element::Type_t::i32);

    const auto& tensor = std::make_shared<ngraph::opset3::Parameter>(inPrecision, ngraph::Shape{1, 800});
    const auto& staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(tensor);
    const auto& dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeNonZero->output(0), staticShapeNonZero->output(1));

    const auto& fnPtr = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{tensor});

    InferenceEngine::CNNNetwork cnnNet(fnPtr);
    for (const auto& outputInfo : cnnNet.getOutputsInfo()) {
        outputInfo.second->setPrecision(ie::Precision::I32);
    }

    ModelPtr model;
    ASSERT_NO_THROW(model = frontEnd->buildInitialModel(cnnNet));
    int numOutputs = 0;
    for (const auto& data : model->datas()) {
        if (data->usage() == DataUsage::Output) {
            numOutputs++;
        }
    }
    ASSERT_EQ(numOutputs, 2);

    const auto& it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->type() == StageType::NonZero;
    });

    ASSERT_NE(it, model->getStages().end());
    const auto& nonZeroStage = *it;

    checkShapeConnection(nonZeroStage->output(1), nonZeroStage->output(0));
}

TEST_F(DSRParsingFromNgraphTests, DSRWithSingleProducerCreatesConnectionBetweenDataAndShape) {
    const auto& inPrecision = ::ngraph::element::Type(::ngraph::element::Type_t::i32);

    const auto& tensor = std::make_shared<ngraph::opset3::Parameter>(inPrecision, ngraph::Shape{800});
    const auto& staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(tensor);
    const auto& dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeNonZero->output(0), staticShapeNonZero->output(1));
    const auto& gatherIndices = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
    const auto& gatherAxis = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});
    const auto& gather = std::make_shared<ngraph::opset3::Gather>(dynamicShapeResolver->output(0), gatherIndices, gatherAxis);

    const auto& fnPtr = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{tensor});

    InferenceEngine::CNNNetwork cnnNet(fnPtr);

    ModelPtr model;
    ASSERT_NO_THROW(model = frontEnd->buildInitialModel(cnnNet));

    const auto& it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->type() == StageType::NonZero;
    });

    ASSERT_NE(it, model->getStages().end());
    const auto& nonZeroStage = *it;

    checkShapeConnection(nonZeroStage->output(1), nonZeroStage->output(0));
}

TEST_F(DSRParsingFromNgraphTests, DSRWithTwoProducersCreatesConnectionBetweenDataAndShape) {
    const auto& inPrecision = ::ngraph::element::Type(::ngraph::element::Type_t::i32);

    const auto& tensor = std::make_shared<ngraph::opset3::Parameter>(inPrecision, ngraph::Shape{800});
    const auto& staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(tensor);
    const auto& reluData = std::make_shared<ngraph::opset3::Relu>(staticShapeNonZero->output(0));
    const auto& reluShape = std::make_shared<ngraph::opset3::Relu>(staticShapeNonZero->output(1));
    const auto& dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
        reluData->output(0), reluShape->output(0));
    const auto& gatherIndices = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
    const auto& gatherAxis = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});
    const auto& gather = std::make_shared<ngraph::opset3::Gather>(dynamicShapeResolver->output(0), gatherIndices, gatherAxis);

    const auto& fnPtr = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{tensor});

    InferenceEngine::CNNNetwork cnnNet(fnPtr);

    ModelPtr model;
    ASSERT_NO_THROW(model = frontEnd->buildInitialModel(cnnNet));

    const auto& it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->type() == StageType::NonZero;
    });

    ASSERT_NE(it, model->getStages().end());
    const auto& nonZeroStage = *it;

    const auto& stageReluData = nonZeroStage->output(0)->singleConsumer();
    const auto& stageReluShape = nonZeroStage->output(1)->singleConsumer();

    checkShapeConnection(stageReluShape->output(0), stageReluData->output(0));
}

} // namespace vpu
