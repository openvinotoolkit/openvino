// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/utility.hpp>

using namespace vpu;

IE_SUPPRESS_DEPRECATED_START

class VPU_SplitLargeKernelConvTest : public GraphTransformerTest {
 protected:
    PassSet pipeline;
    Model model;

 public:
    void InitConvStage(int inputX = 8960, int inputY = 1, bool isOutput4D = true) {
        int kernelx = 16;
        int kernely = 1;
        int kernelStrideX = 1;
        int kernelStrideY = 1;
        int dilationX = 1;
        int dilationY = 1;
        int padx_begin = 7;
        int pady_begin = 0;
        int padx_end = 8;
        int pady_end = 0;
        model = CreateModel();

        auto input = model->addInputData(
            "Input",
            DataDesc(DataType::FP16, DimsOrder::NCHW, {inputX, inputY, 8, 1}));
        model->attrs().set<int>("numInputs", 1);

        Data output;
        if (isOutput4D) {
            output = model->addOutputData(
                "Output",
                DataDesc(DataType::FP16,
                              DimsOrder::NCHW,
                              {(inputX + padx_begin + padx_end - kernelx) / kernelStrideX + 1,
                                (inputY + pady_begin + pady_end - kernely) / kernelStrideY + 1, 8, 1}));
        } else {
            output = model->addOutputData(
                "Output",
                DataDesc(DataType::FP16,
                              DimsOrder::CHW,
                              {(inputX + padx_begin + padx_end - kernelx) / kernelStrideX + 1,
                                (inputY + pady_begin + pady_end - kernely) / kernelStrideY + 1, 8}));
        }

        auto conv = std::make_shared<ie::ConvolutionLayer>(ie::LayerParams{"conv", "Convolution", ie::Precision::FP16});
        conv->_kernel_x = kernelx;
        conv->_kernel_y = kernely;
        conv->_stride_x = kernelStrideX;
        conv->_stride_y = kernelStrideY;
        conv->_dilation_x = dilationX;
        conv->_dilation_x = dilationY;

        conv->_padding.insert(0, padx_begin);
        conv->_padding.insert(1, pady_begin);
        conv->_pads_end.insert(0, padx_end);
        conv->_pads_end.insert(1, pady_end);
        conv->_auto_pad = "same_upper";

        conv->_weights = ie::make_shared_blob<short>({ ie::Precision::FP16, {static_cast<size_t>(kernelx * kernely * 8 * 8)}, ie::Layout::C });
        conv->_weights->allocate();

        frontEnd->parseConvolution(model, conv, {input}, {output});

        pipeline.addPass(passManager->dumpModel("initial"));

        pipeline.addPass(passManager->hwPadding());
        pipeline.addPass(passManager->dumpModel("hwPadding"));

        // if large kernel conv converted to conv that can be ran on HW, then hwConvTiling will work - if not will got an exception
        pipeline.addPass(passManager->splitLargeKernelConv());
        pipeline.addPass(passManager->dumpModel("splitLargeKernelConv"));

        pipeline.addPass(passManager->hwConvTiling());
        pipeline.addPass(passManager->dumpModel("hwConvTiling"));

        pipeline.addPass(passManager->adjustDataLayout());
        pipeline.addPass(passManager->dumpModel("adjustDataLayout"));

        pipeline.addPass(passManager->processSpecialStages());
        pipeline.addPass(passManager->dumpModel("processSpecialStages"));

        pipeline.addPass(passManager->adjustDataLocation());
        pipeline.addPass(passManager->dumpModel("adjustDataLocation"));

        pipeline.addPass(passManager->finalCheck());
    }
};

// Test is going to fail if target convolution is not converted to HW stage
// Conversion to HW stage fails due to #-33366
TEST_F(VPU_SplitLargeKernelConvTest, DISABLED_splitLargeKernelConvIfKernelSizeIs1x16) {
    InitCompileEnv();
    InitConvStage();

    ASSERT_NO_THROW(pipeline.run(model));
}
