// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <myriad_test_case.h>
#include <precision_utils.h>
#include <single_layer_common.hpp>

#include <random>

using namespace vpu;
using namespace ie;

struct param_size {
    size_t x;
    size_t y;
};

using Kernel = param_size;
using Stride = param_size;
using Pad = param_size;

class VPU_SWConvolutionAdaptation : public GraphTransformerTest {
protected:
    PassSet pipeline;
    Model model;

public:
    void InitConvStage(
            DimValues inDims,
            int outChannels,
            Kernel kernel,
            Stride stride,
            Pad pad,
            int group = 1) {
        model = CreateModel();

        const auto input = model->addInputData(
            "Input",
            DataDesc(DataType::FP16, DimsOrder::NCHW, inDims));
        model->attrs().set<int>("numInputs", 1);

        const auto outDims = DimValues{
            {Dim::N, 1},
            {Dim::C, outChannels},
            {Dim::H, (inDims.get(Dim::H, 1) + 2 * pad.y - kernel.y) / stride.y + 1},
            {Dim::W, (inDims.get(Dim::W, 1) + 2 * pad.x - kernel.x) / stride.x + 1}};
        const auto outDesc = DataDesc(DataType::FP16, DimsOrder::NCHW, outDims);
        const auto output = model->addOutputData("Output", outDesc);

        const auto convLayer = std::make_shared<ConvolutionLayer>(
            LayerParams{"conv", "Convolution", Precision::FP16});
        convLayer->_kernel_x = kernel.x;
        convLayer->_kernel_y = kernel.y;
        convLayer->_stride_x = stride.x;
        convLayer->_stride_y = stride.y;
        convLayer->_padding_x = pad.y;
        convLayer->_padding_y = pad.y;
        convLayer->_group = group;

        const auto weightsSize = kernel.x * kernel.y * inDims.get(Dim::C, 1) * outChannels;
        convLayer->_weights = make_shared_blob<short>(
            {Precision::FP16, {weightsSize}, Layout::C});
        convLayer->_weights->allocate();
        smallWeightsRange(convLayer->_weights);

        const auto biasSize = static_cast<size_t>(outChannels);
        convLayer->_biases = make_shared_blob<short>(
            {Precision::FP16, {biasSize}, Layout::C});
        convLayer->_weights->allocate();
        smallWeightsRange(convLayer->_weights);

        frontEnd->parseConvolution(model, convLayer, {input}, {output});

        pipeline.addPass(passManager->dumpModel("initial"));

        pipeline.addPass(passManager->analyzeWeightableLayers());
        pipeline.addPass(passManager->dumpModel("analyzeWeightableLayers"));

        pipeline.addPass(passManager->hwConvTiling());
        pipeline.addPass(passManager->dumpModel("hwConvTiling"));

        pipeline.addPass(passManager->swConvAdaptation());
        pipeline.addPass(passManager->dumpModel("swConvAdaptation"));
    }
private:
    static void smallWeightsRange(const Blob::Ptr& blob) {
        const auto ptr = blob->buffer().as<ie_fp16*>();

        std::mt19937 generator(DEFAULT_SEED_VALUE);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);

        for (size_t count = 0 ; count < blob->size(); ++count) {
            float val = dist(generator) / 512;
            ptr[count] = PrecisionUtils::f32tof16(val);
        }
    }
};

TEST_F(VPU_SWConvolutionAdaptation, rightOrderOfBiasAndScaleStagesWhenUnsuccsessfullTiling) {
    InitCompileEnv();
    const auto inDims = DimValues{{Dim::N, 1},
                                  {Dim::C, 1024},
                                  {Dim::H, 38},
                                  {Dim::W, 38}};
    const int outChannels = 3240;
    const Kernel kernel = {1, 1};
    const Stride stride = {1, 1};
    const Pad pad = {0, 0};
    InitConvStage(inDims, outChannels, kernel, stride, pad);

    ASSERT_NO_THROW(pipeline.run(model));

    const auto& stages = model->getStages();
    for (const auto& stage : stages) {
        if (stage->type() == StageType::Scale) {
            ASSERT_FALSE(std::any_of(stage->nextStages().begin(), stage->nextStages().end(),
                                     [](const Stage& stage) { return stage->type() == StageType::Bias; }));
            ASSERT_TRUE(stage->input(0)->producer()->type() == StageType::Bias);
        }
    }
}
