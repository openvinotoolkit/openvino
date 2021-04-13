// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                  const element::Type &type,
                                                  const std::vector<size_t> &filterSize,
                                                  const std::vector<size_t> &strides,
                                                  const std::vector<ptrdiff_t> &padsBegin,
                                                  const std::vector<ptrdiff_t> &padsEnd,
                                                  const std::vector<size_t> &dilations,
                                                  const op::PadType &autoPad,
                                                  size_t numOutChannels,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t> &outputPadding,
                                                  const std::vector<float> &filterWeights,
                                                  const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {shape[1], numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeConvolutionBackpropData(in, filterWeightsNode, type, strides, padsBegin, padsEnd, dilations, autoPad, addBiases, outputPadding, biasesWeights);
}

std::shared_ptr<Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                  const ngraph::Output<Node> &weights,
                                                  const element::Type &type,
                                                  const std::vector<size_t> &strides,
                                                  const std::vector<ptrdiff_t> &padsBegin,
                                                  const std::vector<ptrdiff_t> &padsEnd,
                                                  const std::vector<size_t> &dilations,
                                                  const op::PadType &autoPad,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t> &outputPadding,
                                                  const std::vector<float> &biasesWeights) {
    auto deconv = std::make_shared<opset1::ConvolutionBackpropData>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<opset1::ConvolutionBackpropData>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad, outputPadding);
    }

    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ngraph::opset1::Add>(deconv, biasesWeightsNode);
        return add;
    } else {
        return deconv;
    }
}

std::shared_ptr<Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                  const ngraph::Output<Node> &outputShape,
                                                  const element::Type &type,
                                                  const std::vector<size_t> &filterSize,
                                                  const std::vector<size_t> &strides,
                                                  const std::vector<ptrdiff_t> &padsBegin,
                                                  const std::vector<ptrdiff_t> &padsEnd,
                                                  const std::vector<size_t> &dilations,
                                                  const op::PadType &autoPad,
                                                  size_t numOutChannels,
                                                  bool addBiases,
                                                  const std::vector<ptrdiff_t> &outputPadding,
                                                  const std::vector<float> &filterWeights,
                                                  const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {shape[1], numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    auto deconv = std::make_shared<opset1::ConvolutionBackpropData>(in, filterWeightsNode, outputShape, strides, padsBegin, padsEnd, dilations, autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<opset1::ConvolutionBackpropData>(in, filterWeightsNode, outputShape, strides, padsBegin,
                                                                   padsEnd, dilations, autoPad, outputPadding);
    }

    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ngraph::opset1::Add>(deconv, biasesWeightsNode);
        return add;
    } else {
        return deconv;
    }
}

std::shared_ptr<Node> makeConvolutionBackpropDataRelaxed(const ngraph::Output<Node> &in,
                                                  const element::Type &weiType,
                                                  const element::Type &outType,
                                                  const std::vector<size_t> &filterSize,
                                                  const std::vector<size_t> &strides,
                                                  const std::vector<ptrdiff_t> &padsBegin,
                                                  const std::vector<ptrdiff_t> &padsEnd,
                                                  const std::vector<size_t> &dilations,
                                                  const op::PadType &autoPad,
                                                  size_t numOutChannels,
                                                  const std::vector<float> &filterWeights) {
    auto inputParamsFP32 = ngraph::builder::makeParams(ngraph::element::f32, { in.get_shape() });
    auto paramOutsFP32 = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParamsFP32));

    auto deconvolutionNodeRelaxed = std::make_shared<op::TypeRelaxed<opset1::ConvolutionBackpropData>>(
            *as_type_ptr<opset1::ConvolutionBackpropData>(ngraph::builder::makeConvolutionBackpropData(
                    paramOutsFP32.front(), ngraph::element::f32, filterSize, strides, padsBegin, padsEnd, dilations, autoPad, numOutChannels)),
            outType);

    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {shape[1], numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(weiType, filterWeightsShape, filterWeights, randomFilterWeights);

    auto newDeconvolution = deconvolutionNodeRelaxed->copy_with_new_inputs(
            {in, filterWeightsNode});

    return newDeconvolution;
}

}  // namespace builder
}  // namespace ngraph