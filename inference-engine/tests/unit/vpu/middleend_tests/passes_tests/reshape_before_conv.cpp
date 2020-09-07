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
        auto convLayer = std::make_shared < ie::ConvolutionLayer > (ie::LayerParams {
            "TestConvolution",
            "StubConv",
            ie::Precision::FP16
        });
        convLayer->_kernel_x = kernel_x;
        convLayer->_kernel_y = kernel_y;
        convLayer->_stride_x = 1;
        convLayer->_stride_y = 1;

        auto weights = InitNewData(
                DimValues{ {Dim::N, output->desc().dim(Dim::C)}, {Dim::C, input->desc().dim(Dim::C)}, {Dim::H, kernel_y}, {Dim::W, kernel_x} },
                "TestConvolution@weights@conv");
        Stage stage = stageBuilder->addConvolutionStage(_model, "TestConvolution", convLayer, input, output,
                weights,
                _model->addFakeData(), _model->addFakeData());

        stage->attrs().set<int>("kernelSizeX", kernel_x);
        stage->attrs().set<int>("kernelSizeY", kernel_y);
        stage->attrs().set<int>("kernelStrideX", 1);
        stage->attrs().set<int>("kernelStrideY", 1);
        stage->attrs().set<bool>("tryHW", true);
    }

    void Validate() {
        const auto datas = _model->datas() | asVector();
        const auto stages = _model->getStages() | asVector();
        ASSERT_EQ(datas.size(), pattern_datas.size());
        ASSERT_EQ(stages.size(), pattern_stages.size());

        for (size_t i = 0; i < datas.size(); i++) {
            ASSERT_EQ((datas[i])->desc(), (pattern_datas[i])->desc());
            ASSERT_EQ((datas[i])->name(), (pattern_datas[i])->name());
        }

        for (size_t i = 0; i < stages.size(); i++) {
            ASSERT_EQ((stages[i])->type(), pattern_stages[i]);
        }
    }

protected:
    PassSet _pipeline;
    Model _model;
    DataVector pattern_datas;
    std::vector<StageType> pattern_stages;
};

enum TwoConvolutionCases : int { NoReshape = 0, ReshapeFirst = 1, ReshapeSecond = 2 };

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
        pattern_datas.push_back(weights);
        pattern_datas.push_back(fake);
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

    void CreateCorrectTwoConvPattern(const Data& input, const Data& mid, const Data& output, TwoConvolutionCases current_case) {
        const auto inDims = input->desc().dims();
        const auto outDims = output->desc().dims();
        const auto midDims = mid->desc().dims();

        const Data fake = _model->addFakeData();
        const Data firstWeights = InitNewData(
                DimValues{ {Dim::N, midDims[Dim::C]}, {Dim::C, inDims[Dim::C]}, {Dim::H, 1}, {Dim::W, 1} },
                "TestConvolution@weights@conv");
        const Data secondWeights = InitNewData(
                DimValues{ {Dim::N, outDims[Dim::C]}, {Dim::C, midDims[Dim::C]}, {Dim::H, 1}, {Dim::W, 1} },
                "TestConvolution@weights@conv");
        const Data inputAfterReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, inDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Input@input-data-after-reshape");
        const Data layerDataBeforeReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, midDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Layer@output-data-before-reshape");
        const Data layerDataAfterReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, midDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Layer@input-data-after-reshape");
        const Data outputBeforeReshape = InitNewData(
                DimValues{ {Dim::N, 1}, {Dim::C, outDims[Dim::C]}, {Dim::H, 8}, {Dim::W, 255} },
                "Output@output-data-before-reshape");

        pattern_datas.push_back(input);
        pattern_datas.push_back(mid);
        pattern_datas.push_back(output);
        pattern_datas.push_back(firstWeights);
        pattern_datas.push_back(fake);
        pattern_datas.push_back(fake);
        pattern_datas.push_back(secondWeights);
        pattern_datas.push_back(fake);
        pattern_datas.push_back(fake);

        switch (current_case) {
        case ReshapeFirst:
            pattern_stages.push_back(StageType::Reshape);
            pattern_stages.push_back(StageType::StubConv);
            pattern_stages.push_back(StageType::Reshape);
            pattern_stages.push_back(StageType::StubConv);

            pattern_datas.push_back(inputAfterReshape);
            pattern_datas.push_back(layerDataBeforeReshape);
            break;
        case ReshapeSecond:
            pattern_stages.push_back(StageType::StubConv);
            pattern_stages.push_back(StageType::Reshape);
            pattern_stages.push_back(StageType::StubConv);
            pattern_stages.push_back(StageType::Reshape);

            pattern_datas.push_back(layerDataAfterReshape);
            pattern_datas.push_back(outputBeforeReshape);
            break;
        case NoReshape:
            pattern_stages.push_back(StageType::StubConv);
            pattern_stages.push_back(StageType::StubConv);
            break;
        }
        // duplicate to pattern additional not connected data
        pattern_datas.push_back(fake);
        pattern_datas.push_back(firstWeights);
        pattern_datas.push_back(secondWeights);
        pattern_datas.push_back(inputAfterReshape);
        pattern_datas.push_back(layerDataBeforeReshape);
        pattern_datas.push_back(layerDataAfterReshape);
        pattern_datas.push_back(outputBeforeReshape);
    }
};

class NoReshapeBeforeConvCases : public ReshapeBeforeConvTests {
protected:
    void CreateIncorrectPattern(const Data& input, const Data& output, size_t kernelx = 1, size_t kernely = 1) {
        const Data fake = _model->addFakeData();
        const Data weights = InitNewData(DimValues{ {Dim::N, output->desc().dim(Dim::C)}, {Dim::C, input->desc().dim(Dim::C)},
                {Dim::H, kernely}, {Dim::W, kernelx} }, "TestConvolution@weights@conv");

        pattern_datas.push_back(input);
        pattern_datas.push_back(output);
        pattern_datas.push_back(weights);
        pattern_datas.push_back(fake);
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

    CreateCorrectPattern(input, output);

    Validate();
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

    InitConvStage(input, output, 3, 5);

    ASSERT_NO_THROW(Compile());

    CreateIncorrectPattern(input, output, 3, 5);

    Validate();
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

    CreateIncorrectPattern(input, output);

    Validate();
}

static const std::vector<DimValues> noPatternDimsForOutput = {
    DimValues{ {Dim::N, 1}, {Dim::C, 12}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 11}, {Dim::H, 224}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 48}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 440}, {Dim::H, 34}, {Dim::W, 60} },
    DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 22}, {Dim::W, 48} },
};

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
    testing::ValuesIn(noPatternDimsForOutput)));

INSTANTIATE_TEST_CASE_P(
        NontargetInputCases, NoReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(noPatternDims),
    testing::ValuesIn(patternOutputDims)));

INSTANTIATE_TEST_CASE_P(
        NontargetCases, NoReshapeBeforeConvCases, testing::Combine(
    testing::ValuesIn(noPatternDims),
    testing::ValuesIn(noPatternDims)));

TEST_F(ReshapeBeforeConvCases, TargetConvolutionBeforeNontarget) {
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

    CreateCorrectTwoConvPattern(input, layerData, output, ReshapeFirst);

    Validate();
}

TEST_F(ReshapeBeforeConvCases, TargetConvolutionAfterNontarget) {
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

    CreateCorrectTwoConvPattern(input, layerData, output, ReshapeSecond);

    Validate();
}

} // namespace vpu
