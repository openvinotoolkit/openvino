// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeConvolution(const ngraph::Output<Node> &in,
                                      const element::Type &type,
                                      const std::vector<size_t> &filterSize,
                                      const std::vector<size_t> &strides,
                                      const std::vector<ptrdiff_t> &padsBegin,
                                      const std::vector<ptrdiff_t> &padsEnd,
                                      const std::vector<size_t> &dilations,
                                      const op::PadType &autoPad,
                                      size_t numOutChannels,
                                      bool addBiases,
                                      const std::vector<float> &filterWeights,
                                      const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, static_cast<size_t>(shape[1].get_length())};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);
    auto conv = std::make_shared<opset1::Convolution>(in, filterWeightsNode, strides, padsBegin, padsEnd, dilations,
                                                      autoPad);
    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {1, numOutChannels , 1, 1}, biasesWeights, randomBiases);
        auto add = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);
        return add;
    } else {
        return conv;
    }
}

std::shared_ptr<Node> makeConvolutionRelaxed(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             const std::vector<size_t> &filterSize,
                                             const std::vector<size_t> &strides,
                                             const std::vector<ptrdiff_t> &padsBegin,
                                             const std::vector<ptrdiff_t> &padsEnd,
                                             const std::vector<size_t> &dilations,
                                             const op::PadType &autoPad,
                                             size_t numOutChannels,
                                             const std::vector<float> &filterWeights) {
    auto inputParamsFP32 = builder::makeDynamicParams(element::f32, { in.get_partial_shape() });
    auto paramOutsFP32 = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParamsFP32));

    auto convolutionNodeRelaxed = std::make_shared<op::TypeRelaxed<opset1::Convolution>>(
            *as_type_ptr<opset1::Convolution>(ngraph::builder::makeConvolution(
                    paramOutsFP32.front(), ngraph::element::f32, filterSize, strides, padsBegin, padsEnd, dilations, autoPad, numOutChannels)),
            element::f32);

    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, static_cast<size_t>(shape[1].get_length())};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    auto newConvolution = convolutionNodeRelaxed->copy_with_new_inputs({in, filterWeightsNode});

    return newConvolution;
}

}  // namespace builder
}  // namespace ngraph
