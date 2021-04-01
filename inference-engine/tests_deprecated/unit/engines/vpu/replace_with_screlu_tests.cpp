// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/stub_stage.hpp>

#include "graph_transformer_tests.hpp"
#include "vpu/model/data_contents/ie_blob_content.hpp"

using namespace vpu;

class VPU_ReplaceWithSCReluTest : public GraphTransformerTest {
protected:
    PassSet pipeline;
    Model model;
public:
    void CreateModelSCRelu() {
        model = CreateModel();

        int inputW = 16;
        int inputH = 16;

        auto input = model->addInputData("Input", DataDesc(DataType::FP16, DimsOrder::NCHW, {inputW, inputH, 2, 1}));
        model->attrs().set<int>("numInputs", 1);

        int kernelx = 16;
        int kernely = 16;
        int kernelStrideX = 1;
        int kernelStrideY = 1;
        int dilationX = 1;
        int dilationY = 1;

        int outW = (inputW - kernelx) / kernelStrideX + 1;
        int outH = (inputH - kernely) / kernelStrideY + 1;

        auto data0 = model->addNewData("data0", DataDesc(DataType::FP16, DimsOrder::NCHW, {outW, outH, 2, 1}));
        auto data1 = model->addNewData("data1", DataDesc(DataType::FP16, DimsOrder::NCHW, {outW, outH, 2, 1}));
        auto data2 = model->addNewData("data2", DataDesc(DataType::FP16, DimsOrder::NCHW, {outW, outH, 4, 1}));
        auto data3 = model->addNewData("data3", DataDesc(DataType::FP16, DimsOrder::NCHW, {outW, outH, 4, 1}));

        auto output = model->addOutputData("Output", DataDesc(DataType::FP16, DimsOrder::NCHW, {outW, outH, 4, 1}));
        model->attrs().set<int>("numOutputs", 1);

        auto scales = model->addConstData("scales", DataDesc(DataType::FP16, DimsOrder::C, {4}));

        float negSlope = -1.0f;

        auto conv = std::make_shared<ie::ConvolutionLayer>(ie::LayerParams{"conv", "StubConv", ie::Precision::FP16});

        conv->_kernel_x = kernelx;
        conv->_kernel_y = kernely;
        conv->_stride_x = kernelStrideX;
        conv->_stride_y = kernelStrideY;
        conv->_dilation_x = dilationX;
        conv->_dilation_x = dilationY;

        conv->_weights = ie::make_shared_blob<short>({ ie::Precision::FP16, {static_cast<size_t>(kernelx * kernely * 2 * 2)}, ie::Layout::C });
        conv->_weights->allocate();

        frontEnd->parseConvolution(model, conv, {input}, {data0});

        stageBuilder->addPowerStage(model,
                                    "Power",
                                    nullptr,
                                    -1.0f,
                                    1.0f,
                                    0.0f,
                                    data0,
                                    data1);

        stageBuilder->addConcatStage(model,
                                    "Concat",
                                    nullptr,
                                    Dim::C,
                                    {data0, data1},
                                    data2);

        stageBuilder->addScaleStage(model,
                                    "ScaleShift",
                                    nullptr,
                                    data2,
                                    scales,
                                    data3);

        stageBuilder->addReLUStage(model,
                                    "Relu",
                                    nullptr,
                                    negSlope,
                                    data3,
                                    output);

        InitPipeline();
    }

    void InitPipeline() {
        pipeline = PassSet();
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->replaceWithSCReLU());
        pipeline.addPass(passManager->dumpModel("replaceWithSCReLU"));
    }
};

TEST_F(VPU_ReplaceWithSCReluTest, CheckStagesReplacement) {
    InitCompileEnv();

    CreateModelSCRelu();
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 2);
    auto stages = model->getStages();
    ASSERT_EQ(stages.back()->type(), StageType::SCRelu);
}
