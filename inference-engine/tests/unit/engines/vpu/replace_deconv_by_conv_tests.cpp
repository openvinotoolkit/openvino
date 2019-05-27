// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/utility.hpp>

#include "graph_transformer_tests.hpp"
using namespace InferenceEngine;
class VPU_ReplaceDeconvByConvTest : public VPU_GraphTransformerTest {
 protected:
    vpu::PassSet pipeline;
    vpu::Model::Ptr model;


 public:
    void InitDeconvStage(
        int kernelx,
        int kernely,
        int inputX=16,
        int inputY=16,
        bool onlySwConvAdaptation = false,
        bool isOutput4D = true) {

        int kernelStrideX = 1;
        int kernelStrideY = 1;
        int dilationX = 1;
        int dilationY = 1;
        model = CreateModel();

        auto input = model->addInputData(
            "Input",
            vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {inputX, inputY, 2, 1}));
        model->attrs().set<int>("numInputs", 1);

        vpu::Data output;
        if (isOutput4D) {
            output = model->addOutputData(
                "Output",
                vpu::DataDesc(vpu::DataType::FP16,
                              vpu::DimsOrder::NCHW,
                              {kernelx + (inputX - 1) * kernelStrideX, kernely + (inputY - 1) * kernelStrideY, 2, 1}));
        } else {
            output = model->addOutputData(
                "Output",
                vpu::DataDesc(vpu::DataType::FP16,
                              vpu::DimsOrder::CHW,
                              {kernelx + (inputX - 1) * kernelStrideX, kernely + (inputY - 1) * kernelStrideY, 2}));
        }

        auto deconv = std::make_shared<DeconvolutionLayer>(LayerParams{"deconv", "Deconvolution", Precision::FP16});
        deconv->_kernel_x = kernelx;
        deconv->_kernel_y = kernely;
        deconv->_stride_x = kernelStrideX;
        deconv->_stride_y = kernelStrideY;
        deconv->_dilation_x = dilationX;
        deconv->_dilation_x = dilationY;

        deconv->_weights = make_shared_blob<short>(Precision::FP16, Layout::C, {static_cast<size_t>(kernelx * kernely * 2 * 2)});
        deconv->_weights->allocate();

        frontEnd->parseDeconvolution(model, deconv, {input}, {output});

        pipeline.addPass(passManager->dumpModel("initial"));

        // if deconv converted to conv than swConvAdaptaion will work - if not will got an exception
        pipeline.addPass(passManager->replaceDeconvByConv());
        pipeline.addPass(passManager->dumpModel("replaceDeconvByConv"));

        pipeline.addPass(passManager->swConvAdaptation());
        pipeline.addPass(passManager->dumpModel("swConvAdaptation"));

        if (!onlySwConvAdaptation) {
            pipeline.addPass(passManager->adjustDataLayout());
            pipeline.addPass(passManager->dumpModel("adjustDataLayout"));

            pipeline.addPass(passManager->processSpecialStages());
            pipeline.addPass(passManager->dumpModel("processSpecialStages"));

            pipeline.addPass(passManager->adjustDataLocation());
            pipeline.addPass(passManager->dumpModel("adjustDataLocation"));

            pipeline.addPass(passManager->finalCheck());
        }
    }
};

TEST_F(VPU_ReplaceDeconvByConvTest, deconvReplacedByConvIfKernelSizeFitsHWUnit) {
    InitCompileEnv();
    InitDeconvStage(15, 15);

    ASSERT_NO_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, deconvCannotBeReplacedByConvIfDisabledInConfig) {
    config.hwBlackList="deconv";
    InitCompileEnv();
    InitDeconvStage(16, 15);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, deconvCannotBeReplacedByConvIfKernelSizeXToBig) {
    InitCompileEnv();
    InitDeconvStage(16, 15);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, deconvCannotBeReplacedByConvIfKernelSizeYToBig) {
    InitCompileEnv();
    InitDeconvStage(15, 16);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, deconvCannotBeReplacedByConvIfOutputNot4D) {
    InitCompileEnv();
    InitDeconvStage(15, 15, 16, 16, false, false);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, canDetectIm2CollBufferFitToBSS) {
    InitCompileEnv();
    // im2col buffer here will be 200 Mbytes, so exception gets thrown , but not in im2coll verification, as shown in next test
    InitDeconvStage(15, 15, 16, 20000);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, canNotDetectIm2CollBufferOverFlow) {
    InitCompileEnv();
    // remaining only sw conv adaptation - since big output might not fit shaves / cmx memory, but we need an im2coll error here
    InitDeconvStage(15, 15, 16, 20000, true);

    ASSERT_NO_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, canDetectIm2CollBufferOverFlow) {
    InitCompileEnv();
    // remaining only sw conv adaptation - since big output might not fit shaves / cmx memory, but we need an im2coll error here
    InitDeconvStage(15, 15, 5016, 20000, true);

    ASSERT_ANY_THROW(pipeline.run(model));
}

TEST_F(VPU_ReplaceDeconvByConvTest, canDetectCMXOrShavesMemoryLimit) {
    InitCompileEnv();
    // remaining only sw conv adaptation - since big output might not fit shaves / cmx memory, but we need an im2coll error here
    InitDeconvStage(15, 15, 16, 20000);

    ASSERT_ANY_THROW(pipeline.run(model));
}
