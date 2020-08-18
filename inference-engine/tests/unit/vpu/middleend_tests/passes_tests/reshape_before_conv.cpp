// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include "ie_memcpy.h"

namespace vpu {

namespace ie = InferenceEngine;

class ReshapeBeforeConvTests : public GraphTransformerTest,
    public testing::WithParamInterface<std::tuple<DimValues, DimValues>> {
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

    Data InitInputData(const DimValues& dims) {
        const auto inDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, dims);
        return _model->addInputData("Input", inDesc);
    }

    Data InitOutputData(const DimValues& dims) {
        const auto outDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, dims);
        return _model->addOutputData("Output", outDesc);
    }

    Data InitNewData(const DimValues& dims, const std::string& name) {
        const auto newDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, dims);
        return _model->addNewData(name, newDesc);
    }

    void InitConvStage(const Data& input, const Data& output, int kernel_x = 1, int kernel_y = 1) {
        auto convLayer = std::make_shared < ie::ConvolutionLayer
                > (ie::LayerParams { "TestConvolution", "StubConv",
                        ie::Precision::FP16 });
        convLayer->_kernel_x = kernel_x;
        convLayer->_kernel_y = kernel_y;
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

    void CompareDatas(const details::ContainerRange<IntrusiveHandleList<DataNode>, false>& datas) {
        ASSERT_EQ(datas.size(), pattern_datas.size());

        auto dataIter = datas.begin();
        for (auto patternIter = pattern_datas.begin(); patternIter != pattern_datas.end();
                std::advance(dataIter, 1),
                std::advance(patternIter, 1)) {
            ASSERT_EQ((*dataIter)->desc(), (*patternIter)->desc());
            ASSERT_EQ((*dataIter)->name(), (*patternIter)->name());
        }
    }

    void CompareStages(const details::ContainerRange<IntrusiveHandleList<StageNode>, false>& stages) {
        ASSERT_EQ(stages.size(), pattern_stages.size());

        auto patternIter = pattern_stages.begin();
        for (auto stageIter = stages.begin(); stageIter != stages.end();
                std::advance(stageIter, 1),
                std::advance(patternIter, 1)) {
            ASSERT_EQ((*stageIter)->type(), *patternIter);
        }
    }

protected:
    PassSet _pipeline;
    Model _model;
    std::vector<Data>pattern_datas;
    std::vector<StageType>pattern_stages;
};

class ReshapeBeforeConvCases : public ReshapeBeforeConvTests {
protected:
    void CreateCorrectPattern(const Data& input, const Data& output) {
        const auto inDims = input->desc().dims();
        const auto outDims = output->desc().dims();

        const Data fake = _model->addFakeData();
        const Data weights = InitNewData(
                DimValues{ {Dim::N, outDims[Dim::C]}, {Dim::C, inDims[Dim::C]}, {Dim::H, 1}, {Dim::W, 1} },
                "TestConvolution@weights@conv");
        const Data inputAfterReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, inDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Input@input-data-after-reshape");
        const Data outputBeforeReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, outDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Output@output-data-before-reshape");

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
    }
};

class NoReshapeBeforeConvCases : public ReshapeBeforeConvTests {
protected:
    void CreateIncorrectPattern(const Data& input, const Data& output, size_t kernelH = 1, size_t kernelW = 1) {
        const Data fake = _model->addFakeData();
        const Data weights = InitNewData(DimValues{ {Dim::N, output->desc().dim(Dim::C)}, {Dim::C, input->desc().dim(Dim::C)},
                {Dim::H, kernelH}, {Dim::W, kernelW} }, "TestConvolution@weights@conv");

        pattern_datas.push_back(input);
        pattern_datas.push_back(output);
        pattern_datas.push_back(fake);
        pattern_datas.push_back(weights);
        pattern_datas.push_back(fake);

        // duplicate to pattern additional not connected data
        pattern_datas.push_back(fake);
        pattern_datas.push_back(weights);

        pattern_stages.push_back(StageType::StubConv);
    }
};

TEST_P(ReshapeBeforeConvCases, CompareCountOfLayersPatternCaseTest) {
    //
    //                          [Fake]              ->|
    //                          [Weights]           ->|
    //                          [Fake]              ->|
    //  [Input] -> (Reshape) -> [InputAfterReshape] ->| -> (StubConv) -> [OutputBeforeReshape] -> (Reshape) -> [Output]
    //
    const auto& p = GetParam();
    const auto& inDims = std::get<0>(p);
    const auto& outDims = std::get<1>(p);

    const auto input = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(outDims);

    InitConvStage(input, output);

    ASSERT_NO_THROW(Compile());

    auto datas = _model->datas();
    const auto stages = _model->getStages();

    CreateCorrectPattern(input, output);

    CompareStages(stages);
    CompareDatas(datas);
}

static const std::vector<DimValues> patternInputDims = {
    DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} }
};
static const std::vector<DimValues> patternOutputDims = {
    DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 490}, {Dim::H, 34}, {Dim::W, 60} }
};

INSTANTIATE_TEST_CASE_P(
        TargetCases, ReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(patternInputDims),
    testing::ValuesIn(patternOutputDims)));

TEST_F(NoReshapeBeforeConvCases, NoChangesForOtherConvKernel) {
    //
    //  [Fake]      ->|
    //  [Weights]   ->|
    //  [Fake]      ->|
    //  [Input]     ->| -> (StubConv) -> [Output]
    //
    DimValues inDims = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues outDims = DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 34}, {Dim::W, 60} };

    const auto input = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(outDims);

    InitConvStage(input, output, 3, 3);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    CreateIncorrectPattern(input, output, 3, 3);

    CompareStages(stages);
    CompareDatas(datas);
}

TEST_P(NoReshapeBeforeConvCases, NoChangesForOtherCases) {
    //
    //  [Fake]      ->|
    //  [Weights]   ->|
    //  [Fake]      ->|
    //  [Input]     ->| -> (StubConv) -> [Output]
    //
    const auto& p = GetParam();
    const auto& inDims = std::get<0>(p);
    const auto& outDims = std::get<1>(p);

    const auto input = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 1);
    const auto output = InitOutputData(outDims);

    InitConvStage(input, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    CreateIncorrectPattern(input, output);

    CompareStages(stages);
    CompareDatas(datas);
}

static const std::vector<DimValues> noPatternDims = {
    DimValues{ {Dim::N, 1}, {Dim::C, 12}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 8}, {Dim::W, 64} },
    DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 82}, {Dim::W, 255} },
    DimValues{ {Dim::N, 1}, {Dim::C, 490}, {Dim::H, 72}, {Dim::W, 90} },
    DimValues{ {Dim::N, 1}, {Dim::C, 490}, {Dim::H, 64}, {Dim::W, 30} },
    DimValues{ {Dim::N, 1}, {Dim::C, 11}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 48}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 440}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 22}, {Dim::W, 48} },
};

INSTANTIATE_TEST_CASE_P(
        NontargetOutputCases, NoReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(patternInputDims),
    testing::ValuesIn(noPatternDims)));

INSTANTIATE_TEST_CASE_P(
        NontargetInputCases, NoReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(noPatternDims),
    testing::ValuesIn(patternOutputDims)));

INSTANTIATE_TEST_CASE_P(
        NontargetCases, NoReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(noPatternDims),
    testing::ValuesIn(noPatternDims)));

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
    DimValues inDims = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues firstOutDims = DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues secondOutDims = DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 34}, {Dim::W, 60} };

    const auto firstInput = InitInputData(inDims);
    const auto secondInput = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 2);
    const auto firstOutput = InitOutputData(firstOutDims);
    const auto secondOutput = InitOutputData(secondOutDims);

    InitConvStage(firstInput, firstOutput);
    InitConvStage(secondInput, secondOutput);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(
            DimValues{ {Dim::N, 10}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(
            DimValues{ {Dim::N, 128}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data firstInputAfterReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 8}, {Dim::W, 255} },
            "Input@input-data-after-reshape");
    const Data secondInputAfterReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 8}, {Dim::W, 255} },
            "Input@input-data-after-reshape");
    const Data firstOutputBeforeReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 8}, {Dim::W, 255} },
            "Output@output-data-before-reshape");
    const Data secondOutputBeforeReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 8}, {Dim::W, 255} },
            "Output@output-data-before-reshape");

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

    CompareStages(stages);
    CompareDatas(datas);
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
    DimValues InDims = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues firstOutDims = DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues secondOutDims = DimValues{ {Dim::N, 1}, {Dim::C, 222}, {Dim::H, 34}, {Dim::W, 60} };

    const auto firstInput = InitInputData(InDims);
    const auto secondInput = InitInputData(InDims);
    _model->attrs().set<int>("numInputs", 2);
    const auto firstOutput = InitOutputData(firstOutDims);
    const auto secondOutput = InitOutputData(secondOutDims);

    InitConvStage(firstInput, firstOutput);
    InitConvStage(secondInput, secondOutput);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(
            DimValues { {Dim::N, 10}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(
            DimValues { {Dim::N, 222}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data inputAfterReshape = InitNewData(
            DimValues { {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 8}, {Dim::W, 255} },
            "Input@input-data-after-reshape");
    const Data outputBeforeReshape = InitNewData(
            DimValues { {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 8}, {Dim::W, 255} },
            "Output@output-data-before-reshape");

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

    CompareStages(stages);
    CompareDatas(datas);
}

TEST_F(ReshapeBeforeConvTests, TargetConvolutionBeforeNontarget) {
    //
    //                          [Fake]              ->|                                                           [Fake]          ->|
    //                          [FirstWeights]      ->|                                                           [SecondWeights] ->|
    //                          [Fake]              ->|                                                           [Fake]          ->|
    //  [Input] -> (Reshape) -> [InputAfterReshape] ->| -> (StubConv) -> [LayerDataBeforeReshape] -> (Reshape) -> [LayerData]     ->| ->
    //
    //  -> (StubConv) -> [Output]
    //
    DimValues inDims = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues layerDims = DimValues{ {Dim::N, 1}, {Dim::C, 490}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues outDims = DimValues{ {Dim::N, 1}, {Dim::C, 10}, {Dim::H, 34}, {Dim::W, 60} };

    const auto input = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 1);
    const auto layerData = InitNewData(layerDims, "Layer");
    const auto output = InitOutputData(outDims);

    InitConvStage(input, layerData);
    InitConvStage(layerData, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(
            DimValues{ {Dim::N, 490}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(
            DimValues{ {Dim::N, 10}, {Dim::C, 490}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data inputAfterReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 8}, {Dim::W, 255} },
            "Input@input-data-after-reshape");
    const Data layerDataBeforeReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 490}, {Dim::H, 8}, {Dim::W, 255} },
            "Layer@output-data-before-reshape");

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

    CompareStages(stages);
    CompareDatas(datas);
}

TEST_F(ReshapeBeforeConvTests, TargetConvolutionAfterNontarget) {
    //
    //  [Fake]         ->|                                              [Fake]                  ->|
    //  [FirstWeights] ->|                                              [SecondWeights]         ->|
    //  [Fake]         ->|                                              [Fake]                  ->|
    //  [Input]        ->| -> (StubConv) -> [LayerData] -> (Reshape) -> [LayerDataAfterReshape] ->| ->
    //
    //  -> (StubConv) -> [OutputBeforeReshape] -> (Reshape) -> [Output]
    //
    DimValues inDims = DimValues{ {Dim::N, 1}, {Dim::C, 800}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues layerDims = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };
    DimValues outDims = DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 34}, {Dim::W, 60} };

    const auto input = InitInputData(inDims);
    _model->attrs().set<int>("numInputs", 1);
    const auto layerData = InitNewData(layerDims, "Layer");
    const auto output = InitOutputData(outDims);

    InitConvStage(input, layerData);
    InitConvStage(layerData, output);

    ASSERT_NO_THROW(Compile());

    const auto datas = _model->datas();
    const auto stages = _model->getStages();

    const Data fake = _model->addFakeData();
    const Data firstWeights = InitNewData(
            DimValues{ {Dim::N, 608}, {Dim::C, 800}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data secondWeights = InitNewData(
            DimValues{ {Dim::N, 128}, {Dim::C, 608}, {Dim::H, 1}, {Dim::W, 1} },
            "TestConvolution@weights@conv");
    const Data layerDataAfterReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 8}, {Dim::W, 255} },
            "Layer@input-data-after-reshape");
    const Data outputBeforeReshape = InitNewData(
            DimValues{ {Dim::N, 1}, {Dim::C, 128}, {Dim::H, 8}, {Dim::W, 255} },
            "Output@output-data-before-reshape");

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

    CompareStages(stages);
    CompareDatas(datas);
}

} // namespace vpu
