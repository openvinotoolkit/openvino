// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <tuple>
#include <set>

#include <ie_layers_internal.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>

namespace vpu {

void FrontEnd::parseConvolution(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto input = inputs[0];
    auto output = outputs[0];

    if (!(input->desc().numDims() == 3 || input->desc().numDims() == 4)) {
        VPU_THROW_EXCEPTION << "Convolution supports only 3D or 4D input";
    }
    if (output->desc().numDims() != input->desc().numDims()) {
        VPU_THROW_EXCEPTION << "Convolution supports only same num dims in input and output";
    }

    //
    // Extract parameters
    //

    auto convLayer = std::dynamic_pointer_cast<ie::ConvolutionLayer>(layer);
    IE_ASSERT(convLayer != nullptr);

    int kernelSizeX = convLayer->_kernel_x;
    int kernelSizeY = convLayer->_kernel_y;

    int kernelStrideX = convLayer->_stride_x;
    int kernelStrideY = convLayer->_stride_y;

    auto paddings = getPaddings(*convLayer);
    int padLeft = paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop = paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : padTop;

    int dilationX = convLayer->_dilation_x;
    int dilationY = convLayer->_dilation_y;

    int groupSize = convLayer->_group;

    //
    // Check if HW is applicable
    //

    auto tryHW = env.config.hwOptimization;

    if (kernelStrideX != kernelStrideY) {
        tryHW = false;
    }

    // TODO: support dilated convolution
    if ((dilationX != 1 || dilationY != 1) && (!env.config.hwDilation)) {
        tryHW = false;
    }

    if (kernelSizeX > 15 || kernelSizeY > 15 || kernelStrideX > 8) {
        tryHW = false;
    }

    if (env.netConfig.hwDisabled(layer->name)) {
        tryHW = false;
    }

    if (output->desc().numDims() < 4) {
        tryHW = false;
    }

    //
    // Create const datas
    //

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    IE_ASSERT(weights->desc().totalDimSize() >=
              kernelSizeX * kernelSizeY * (input->desc().dim(Dim::C) / groupSize) * output->desc().dim(Dim::C));
    weights = model->duplicateData(
        weights,
        "@conv",
        DataDesc({
            kernelSizeX,
            kernelSizeY,
            input->desc().dim(Dim::C) / groupSize,
            output->desc().dim(Dim::C)}));

    if (biases->usage() != DataUsage::Fake) {
        IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
        biases = model->duplicateData(
            biases,
            "@conv",
            DataDesc({output->desc().dim(Dim::C)}));
    }

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        layer->name,
        StageType::StubConv,
        layer,
        {input, weights, biases},
        {output});

    stage->attrs().set<int>("kernelSizeX", kernelSizeX);
    stage->attrs().set<int>("kernelSizeY", kernelSizeY);

    stage->attrs().set<int>("kernelStrideX", kernelStrideX);
    stage->attrs().set<int>("kernelStrideY", kernelStrideY);

    stage->attrs().set<int>("padLeft", padLeft);
    stage->attrs().set<int>("padRight", padRight);
    stage->attrs().set<int>("padTop", padTop);
    stage->attrs().set<int>("padBottom", padBottom);

    stage->attrs().set<int>("dilationX", dilationX);
    stage->attrs().set<int>("dilationY", dilationY);

    stage->attrs().set<int>("groupSize", groupSize);

    stage->attrs().set<bool>("tryHW", tryHW);
}

}  // namespace vpu
