// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                       const element::Type &type,
                                                       const std::vector<size_t> &filterSize,
                                                       const std::vector<size_t> &strides,
                                                       const std::vector<ptrdiff_t> &padsBegin,
                                                       const std::vector<ptrdiff_t> &padsEnd,
                                                       const std::vector<size_t> &dilations,
                                                       const op::PadType &autoPad,
                                                       size_t numOutChannels,
                                                       size_t numGroups,
                                                       bool addBiases,
                                                       const std::vector<ptrdiff_t> &outputPadding,
                                                       const std::vector<float> &filterWeights,
                                                       const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {static_cast<size_t>(shape[1].get_length()), numOutChannels};
    if (filterWeightsShape[0] % numGroups || filterWeightsShape[1] % numGroups)
        throw std::runtime_error("incorrect shape for GroupConvolutionBackpropData");
    filterWeightsShape[0] /= numGroups;
    filterWeightsShape[1] /= numGroups;
    filterWeightsShape.insert(filterWeightsShape.begin(), numGroups);
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeGroupConvolutionBackpropData(in, filterWeightsNode, type, strides, padsBegin, padsEnd, dilations, autoPad, addBiases,
                                            outputPadding, biasesWeights);
}

std::shared_ptr<Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
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
    auto deconv = std::make_shared<opset1::GroupConvolutionBackpropData>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<opset1::GroupConvolutionBackpropData>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad, outputPadding);
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

std::shared_ptr<Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                       const ngraph::Output<Node> &outputShape,
                                                       const element::Type &type,
                                                       const std::vector<size_t> &filterSize,
                                                       const std::vector<size_t> &strides,
                                                       const std::vector<ptrdiff_t> &padsBegin,
                                                       const std::vector<ptrdiff_t> &padsEnd,
                                                       const std::vector<size_t> &dilations,
                                                       const op::PadType &autoPad,
                                                       size_t numOutChannels,
                                                       size_t numGroups,
                                                       bool addBiases,
                                                       const std::vector<ptrdiff_t> &outputPadding,
                                                       const std::vector<float> &filterWeights,
                                                       const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {static_cast<size_t>(shape[1].get_length()), numOutChannels};
    if (filterWeightsShape[0] % numGroups || filterWeightsShape[1] % numGroups)
        throw std::runtime_error("incorrect shape for GroupConvolutionBackpropData");
    filterWeightsShape[0] /= numGroups;
    filterWeightsShape[1] /= numGroups;
    filterWeightsShape.insert(filterWeightsShape.begin(), numGroups);
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    auto deconv = std::make_shared<opset1::GroupConvolutionBackpropData>(in, filterWeightsNode, outputShape, strides, padsBegin, padsEnd, dilations, autoPad);

    if (!outputPadding.empty()) {
        deconv = std::make_shared<opset1::GroupConvolutionBackpropData>(in, filterWeightsNode, outputShape, strides, padsBegin,
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

}  // namespace builder
}  // namespace ngraph