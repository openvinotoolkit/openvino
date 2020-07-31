// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include "ie_memcpy.h"

namespace vpu {

namespace ie = InferenceEngine;

class ReshapeBeforeConvTests : public GraphTransformerTest,
    public testing::WithParamInterface<size_t> {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
        ASSERT_NO_FATAL_FAILURE(InitPipeline());

        _model = CreateModel();
    }

    void Compile() {
        _pipeline.run(_model);
    }

    void InitPipeline() {
        _pipeline = PassSet();
        _pipeline.addPass(passManager->dumpModel("before-convert-shape-notation"));
        _pipeline.addPass(passManager->initialCheck());
        _pipeline.addPass(passManager->reshapeBeforeConvTiling());
        _pipeline.addPass(passManager->dumpModel("after-convert-shape-notation"));
    }

    Data InitInputData(int inputW, int inputH, int inputC, int inputN) {
        const auto inDims = DimValues{
            {Dim::N, inputN},
            {Dim::C, inputC},
            {Dim::H, inputH},
            {Dim::W, inputW}};
        const auto inDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, inDims);
        return _model->addInputData("Input", inDesc);
    }

    Data InitOutputData(int outputW, int outputH, int outputC, int outputN) {
        const auto outDims = DimValues{
            {Dim::N, outputN},
            {Dim::C, outputC},
            {Dim::H, outputH},
            {Dim::W, outputW}};
        const auto outDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, outDims);
        return _model->addOutputData("Output", outDesc);
    }

    Data InitNewData(int dimW, int dimH, int dimC, int dimN, const std::string& name) {
        const auto newDims = DimValues{
                {Dim::N, dimN},
                {Dim::C, dimC},
                {Dim::H, dimH},
                {Dim::W, dimW}};
        const auto newDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, newDims);
        return _model->addNewData(name, newDesc);
    }

    void InitConvStage(const Data& input, const Data& output) {
        auto convLayer = std::make_shared < ie::ConvolutionLayer
                > (ie::LayerParams { "TestConvolution", "StubConv",
                        ie::Precision::FP16 });
        convLayer->_kernel_x = 1;
        convLayer->_kernel_y = 1;
        convLayer->_stride_x = 1;
        convLayer->_stride_y = 1;
        convLayer->_padding_x = 0;
        convLayer->_padding_y = 0;

        convLayer->_weights = ie::make_shared_blob<short>({
            ie::Precision::FP16, {static_cast<size_t>(convLayer->_kernel_x * convLayer->_kernel_y *
            input->desc().dim(Dim::C) * output->desc().dim(Dim::C))}, ie::Layout::C });
        convLayer->_weights->allocate();

        frontEnd->parseConvolution(_model, convLayer, { input }, { output });
    }

    void CompareDatas(const details::ContainerRange<IntrusiveHandleList<DataNode>, false>& datas,
            const std::vector<Data>& pattern) {
        ASSERT_EQ(datas.size(), pattern.size());

        auto dataIter = datas.begin();
        for (auto patternIter = pattern.begin(); patternIter != pattern.end();
                std::advance(dataIter, 1),
                std::advance(patternIter, 1)) {
            ASSERT_EQ((*dataIter)->desc(), (*patternIter)->desc());
            ASSERT_EQ((*dataIter)->name(), (*patternIter)->name());
        }
    }

    void CompareStages(const details::ContainerRange<IntrusiveHandleList<StageNode>, false>& stages, const std::vector<StageType>& pattern) {
        ASSERT_EQ(stages.size(), pattern.size());

        auto patternIter = pattern.begin();
        for (auto stageIter = stages.begin(); stageIter != stages.end();
                std::advance(stageIter, 1),
                std::advance(patternIter, 1)) {
            ASSERT_EQ((*stageIter)->type(), *patternIter);
        }
    }

protected:
    PassSet _pipeline;
    Model _model;
};

TEST_P(ReshapeBeforeConvTests, CompareCountOfLayersPatternCaseTest) {
    //
    //                          [Fake]              ->|
    //                          [Weights]           ->|
    //                          [Fake]              ->|
    //  [Input] -> (Reshape) -> [InputAfterReshape] ->| -> (StubConv) -> [OutputBeforeReshape] -> (Reshape) -> [Output]
    //
    const auto& p = GetParam();

    const auto input = InitInputData(60, 34, 608, 1);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(60, 34, p, 1);

    InitConvStage(input, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data weights = InitNewData(1, 1, 608, p, "TestConvolution@weights@conv");
    const Data inputAfterReshape = InitNewData(255, 8, 608, 1, "Input@input-data-after-reshape");
    const Data outputBeforeReshape = InitNewData(255, 8, p, 1, "Output@output-data-before-reshape");

    pattern_datas.push_back(input);
    pattern_datas.push_back(output);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

static const ie::SizeVector patternDims = {
    10,
    128,
    490
};

INSTANTIATE_TEST_CASE_P(
        accuracy, ReshapeBeforeConvTests,
        ::testing::ValuesIn(patternDims));

TEST_F(ReshapeBeforeConvTests, NoChangesForOtherChannel) {
    //
    //  [Fake]      ->|
    //  [Weights]   ->|
    //  [Fake]      ->|
    //  [Input]     ->| -> (StubConv) -> [Output]
    //

    const auto input = InitInputData(60, 34, 608, 1);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(60, 34, 111, 1);

    InitConvStage(input, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data weights = InitNewData(1, 1, 608, 111, "TestConvolution@weights@conv");

    pattern_datas.push_back(input);
    pattern_datas.push_back(output);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);
    pattern_datas.push_back(fake);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);

    pattern_stages.push_back(StageType::StubConv);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

TEST_F(ReshapeBeforeConvTests, NoChangesForOtherDimensions) {
    //
    //  [Fake]      ->|
    //  [Weights]   ->|
    //  [Fake]      ->|
    //  [Input]     ->| -> (StubConv) -> [Output]
    //

    const auto input = InitInputData(30, 68, 608, 1);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(30, 68, 10, 1);

    InitConvStage(input, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data weights = InitNewData(1, 1, 608, 10, "TestConvolution@weights@conv");

    pattern_datas.push_back(input);
    pattern_datas.push_back(output);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);
    pattern_datas.push_back(fake);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(weights);

    pattern_stages.push_back(StageType::StubConv);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

TEST_F(ReshapeBeforeConvTests, TwoTargetNotConnectedConvolutions) {
    //
    //                               [Fake]                   ->|
    //                               [FirstWeights]           ->|
    //                               [Fake]                   ->|
    //  [FirstInput] -> (Reshape) -> [FirstInputAfterReshape] ->| -> (StubConv) -> [FirstOutputBeforeReshape] -> (Reshape) -> [FirstOutput]
    //
    //                                [Fake]                    ->|
    //                                [SecondWeights]           ->|
    //                                [Fake]                    ->|
    //  [SecondInput] -> (Reshape) -> [SecondInputAfterReshape] ->| -> (StubConv) -> [SecondOutputBeforeReshape] -> (Reshape) -> [SecondOutput]
    //

    const auto firstInput = InitInputData(60, 34, 608, 1);
    const auto secondInput = InitInputData(60, 34, 608, 1);
    _model->attrs().set<int>("numInputs", 2);
    const auto firstOutput = InitOutputData(60, 34, 10, 1);
    const auto secondOutput = InitOutputData(60, 34, 128, 1);

    InitConvStage(firstInput, firstOutput);
    InitConvStage(secondInput, secondOutput);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(1, 1, 608, 10, "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(1, 1, 608, 128, "TestConvolution@weights@conv");
    const Data firstInputAfterReshape = InitNewData(255, 8, 608, 1, "Input@input-data-after-reshape");
    const Data secondInputAfterReshape = InitNewData(255, 8, 608, 1, "Input@input-data-after-reshape");
    const Data firstOutputBeforeReshape = InitNewData(255, 8, 10, 1, "Output@output-data-before-reshape");
    const Data secondOutputBeforeReshape = InitNewData(255, 8, 128, 1, "Output@output-data-before-reshape");

    pattern_datas.push_back(firstInput);
    pattern_datas.push_back(secondInput);
    pattern_datas.push_back(firstOutput);
    pattern_datas.push_back(secondOutput);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstInputAfterReshape);
    pattern_datas.push_back(firstOutputBeforeReshape);
    pattern_datas.push_back(secondInputAfterReshape);
    pattern_datas.push_back(secondOutputBeforeReshape);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(firstInputAfterReshape);
    pattern_datas.push_back(secondInputAfterReshape);
    pattern_datas.push_back(firstOutputBeforeReshape);
    pattern_datas.push_back(secondOutputBeforeReshape);

    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

TEST_F(ReshapeBeforeConvTests, OneTargetAndOneNontargetNotConnectedConvolutions) {
    //
    //                               [Fake]                   ->|
    //                               [FirstWeights]           ->|
    //                               [Fake]                   ->|
    //  [FirstInput] -> (Reshape) -> [InputAfterReshape] ->| -> (StubConv) -> [OutputBeforeReshape] -> (Reshape) -> [FirstOutput]
    //
    //  [Fake]            ->|
    //  [SecondWeights]   ->|
    //  [Fake]            ->|
    //  [SecondInput]     ->| -> (StubConv) -> [SecondOutput]
    //

    const auto firstInput = InitInputData(60, 34, 608, 1);
    const auto secondInput = InitInputData(60, 34, 608, 1);
    _model->attrs().set<int>("numInputs", 2);
    const auto firstOutput = InitOutputData(60, 34, 10, 1);
    const auto secondOutput = InitOutputData(60, 34, 222, 1);

    InitConvStage(firstInput, firstOutput);
    InitConvStage(secondInput, secondOutput);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(1, 1, 608, 10, "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(1, 1, 608, 222, "TestConvolution@weights@conv");
    const Data inputAfterReshape = InitNewData(255, 8, 608, 1, "Input@input-data-after-reshape");
    const Data outputBeforeReshape = InitNewData(255, 8, 10, 1, "Output@output-data-before-reshape");

    pattern_datas.push_back(firstInput);
    pattern_datas.push_back(secondInput);
    pattern_datas.push_back(firstOutput);
    pattern_datas.push_back(secondOutput);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

TEST_F(ReshapeBeforeConvTests, TargetConvolutionBeforeNontarger) {
    //
    //                          [Fake]              ->|                                                           [Fake]          ->|
    //                          [FirstWeights]      ->|                                                           [SecondWeights] ->|
    //                          [Fake]              ->|                                                           [Fake]          ->|
    //  [Input] -> (Reshape) -> [InputAfterReshape] ->| -> (StubConv) -> [LayerDataBeforeReshape] -> (Reshape) -> [LayerData]     ->| ->
    //
    //  -> (StubConv) -> [Output]
    //

    const auto input = InitInputData(60, 34, 608, 1);
    _model->attrs().set<int>("numInputs", 1);
    const auto layerData = InitNewData(60, 34, 490, 1, "Layer");
    const auto output = InitOutputData(60, 34, 10, 1);

    InitConvStage(input, layerData);
    InitConvStage(layerData, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(1, 1, 608, 490, "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(1, 1, 490, 10, "TestConvolution@weights@conv");
    const Data inputAfterReshape = InitNewData(255, 8, 608, 1, "Input@input-data-after-reshape");
    const Data layerDataBeforeReshape = InitNewData(255, 8, 490, 1, "Layer@output-data-before-reshape");

    pattern_datas.push_back(input);
    pattern_datas.push_back(layerData);
    pattern_datas.push_back(output);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(layerDataBeforeReshape);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(inputAfterReshape);
    pattern_datas.push_back(layerDataBeforeReshape);

    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

TEST_F(ReshapeBeforeConvTests, TargetConvolutionAfterNontarger) {
    //
    //  [Fake]         ->|                                              [Fake]                  ->|
    //  [FirstWeights] ->|                                              [SecondWeights]         ->|
    //  [Fake]         ->|                                              [Fake]                  ->|
    //  [Input]        ->| -> (StubConv) -> [LayerData] -> (Reshape) -> [LayerDataAfterReshape] ->| ->
    //
    //  -> (StubConv) -> [OutputBeforeReshape] -> (Reshape) -> [Output]
    //

    const auto input = InitInputData(60, 34, 800, 1);
    _model->attrs().set<int>("numInputs", 1);
    const auto layerData = InitNewData(60, 34, 608, 1, "Layer");
    const auto output = InitOutputData(60, 34, 128, 1);

    InitConvStage(input, layerData);
    InitConvStage(layerData, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();
    auto pattern_datas = std::vector<Data>();
    auto pattern_stages = std::vector<StageType>();

    // this adds to model additional not connected data
    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(1, 1, 800, 608, "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(1, 1, 608, 128, "TestConvolution@weights@conv");
    const Data layerDataAfterReshape = InitNewData(255, 8, 608, 1, "Layer@input-data-after-reshape");
    const Data outputBeforeReshape = InitNewData(255, 8, 128, 1, "Output@output-data-before-reshape");

    pattern_datas.push_back(input);
    pattern_datas.push_back(layerData);
    pattern_datas.push_back(output);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(fake);
    pattern_datas.push_back(layerDataAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    // duplicate to pattern additional not connected data
    pattern_datas.push_back(fake);
    pattern_datas.push_back(firstWeights);
    pattern_datas.push_back(secondWeights);
    pattern_datas.push_back(layerDataAfterReshape);
    pattern_datas.push_back(outputBeforeReshape);

    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);
    pattern_stages.push_back(StageType::StubConv);
    pattern_stages.push_back(StageType::Reshape);

    CompareStages(stages, pattern_stages);
    CompareDatas(datas, pattern_datas);
}

} // namespace vpu
