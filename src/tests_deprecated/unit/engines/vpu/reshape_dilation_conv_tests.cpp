// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_ReshapeDilationConvTest: public GraphTransformerTest {
protected:
    PassSet pipeline;
    Model model;

public:
    void InitDilationConvStage(int kernelx, int kernely, int dilationX,
            int dilationY, int padX, int padY, int inputX = 16, int inputY = 16,
            bool onlySwConvAdaptation = false) {

        int kernelStrideX = 1;
        int kernelStrideY = 1;
        model = CreateModel();

        auto input = model->addInputData("Input",
                DataDesc(DataType::FP16, DimsOrder::NCHW,
                        { inputX, inputY, 2, 1 }));
        model->attrs().set<int>("numInputs", 1);

        auto padx_new = padX - (kernelx - 1) * (dilationX - 1) / 2;
        auto pady_new = padY - (kernely - 1) * (dilationY - 1) / 2;
        Data output;

        output = model->addOutputData("Output",
                DataDesc(DataType::FP16, DimsOrder::NCHW,
                        { (inputX + 2 * padx_new - kernelx) / kernelStrideX + 1,
                                (inputY + 2 * pady_new - kernely)
                                        / kernelStrideY + 1, 2, 1 }));

        auto dilationconv = std::make_shared < ie::ConvolutionLayer
                > (ie::LayerParams { "dilationconv", "StubConv",
                        ie::Precision::FP16 });
        dilationconv->_kernel_x = kernelx;
        dilationconv->_kernel_y = kernely;
        dilationconv->_stride_x = kernelStrideX;
        dilationconv->_stride_y = kernelStrideY;
        dilationconv->_dilation_x = dilationX;
        dilationconv->_dilation_y = dilationY;
        dilationconv->_padding_x = padX;
        dilationconv->_padding_y = padY;

        dilationconv->_weights = ie::make_shared_blob<short>(
                { ie::Precision::FP16, { static_cast<size_t>(kernelx * kernely
                        * 2 * 2) }, ie::Layout::C });
        dilationconv->_weights->allocate();

        frontEnd->parseConvolution(model, dilationconv, { input }, { output });

        pipeline.addPass(passManager->dumpModel("initial"));

        pipeline.addPass(passManager->reshapeDilationConv());
        pipeline.addPass(passManager->dumpModel("reshapeDilationConv"));

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

TEST_F(VPU_ReshapeDilationConvTest, reshapeDilationConvIfFitsHWUnit) {
    InitCompileEnv();
    InitDilationConvStage(3, 3, 2, 2, 1, 1);

    ASSERT_NO_THROW(pipeline.run(model));
}
