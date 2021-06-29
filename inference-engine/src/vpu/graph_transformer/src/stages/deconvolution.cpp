// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <tuple>
#include <set>

#include <legacy/ie_layers_internal.hpp>

#include <vpu/stages/stub_stage.hpp>

namespace vpu {

void FrontEnd::parseDeconvolution(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);
    auto deconv = ngraph::as_type_ptr<ngraph::opset4::ConvolutionBackpropData>(node);
    VPU_THROW_UNLESS(deconv != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", deconv->get_friendly_name(), deconv->get_type_name());
    //
    // Extract parameters
    //
    auto filtersShape = deconv->input_value(1).get_partial_shape().to_shape();
    auto strides = deconv->get_strides();
    auto padsBegin = deconv->get_pads_begin();
    auto padsEnd = deconv->get_pads_end();
    auto dilations = deconv->get_dilations();
    size_t group = deconv->input_value(1).get_shape()[0];

    int kernelSizeX = filtersShape[0]; //deconvLayer->_kernel_x;
    int kernelSizeY = filtersShape[1];

    int kernelStrideX = strides[0];  //deconvLayer->_stride_x;
    int kernelStrideY = strides[1];  //deconvLayer->_stride_y;

    // auto paddings = getPaddings(*deconvLayer);
    int padLeft = padsBegin[0]; // paddings.begin.exist(ie::X_AXIS) ? paddings.begin[ie::X_AXIS] : 0;
    int padRight = padsEnd[0];  // paddings.end.exist(ie::X_AXIS) ? paddings.end[ie::X_AXIS] : padLeft;
    int padTop =  padsBegin[1]; // paddings.begin.exist(ie::Y_AXIS) ? paddings.begin[ie::Y_AXIS] : 0;
    int padBottom = padsEnd[1]; // paddings.end.exist(ie::Y_AXIS) ? paddings.end[ie::Y_AXIS] : 0;

    int dilationX = dilations[0]; // deconvLayer->_dilation_x;
    int dilationY = dilations[1]; // deconvLayer->_dilation_y;

    int groupSize = group; // deconvLayer->_group;

    //
    // Create const datas
    //

    auto input = inputs[0];
    auto output = outputs[0];

    if ((groupSize == 0) ||
        (groupSize > input->desc().dim(Dim::C)) ||
        (input->desc().dim(Dim::C) % groupSize != 0) ||
        (groupSize > output->desc().dim(Dim::C)) ||
        (output->desc().dim(Dim::C) % groupSize != 0)) {
        VPU_THROW_EXCEPTION << "DeconvolutionLayer has invalid group value";
    }

    Data weights, biases;
    const auto weightsNode = deconv->input_value(1).get_node_shared_ptr();
    const auto biasNode = deconv->inputs().size() == 3 ? deconv->input_value(2).get_node_shared_ptr() : NodePtr();
    std::tie(weights, biases) = getWeightsAndBiases(model, deconv->get_friendly_name(), weightsNode, biasNode);

    IE_ASSERT(weights->desc().totalDimSize() >=
              kernelSizeX * kernelSizeY * (input->desc().dim(Dim::C) / groupSize) * output->desc().dim(Dim::C));
    weights = model->duplicateData(
        weights,
        "@deconv",
        DataDesc({
            kernelSizeX,
            kernelSizeY,
            input->desc().dim(Dim::C) / groupSize,
            output->desc().dim(Dim::C)}));

    if (biases->usage() != DataUsage::Fake) {
        IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
        biases = model->duplicateData(
            biases,
            "@deconv",
            DataDesc({output->desc().dim(Dim::C)}));
    }

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        deconv->get_friendly_name(),
        StageType::StubDeconv,
        deconv,
        {input, weights, biases, model->addFakeData()},
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
    stage->attrs().set<bool>("tryHW", true);
}

}  // namespace vpu
