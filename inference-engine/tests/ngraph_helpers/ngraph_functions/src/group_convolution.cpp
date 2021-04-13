// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGroupConvolution(const ngraph::Output<Node> &in,
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
                                      const std::vector<float> &filterWeights,
                                      const std::vector<float> &biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, shape[1]};
    if (filterWeightsShape[0] % numGroups || filterWeightsShape[1] % numGroups)
        throw std::runtime_error("incorrected shape for GroupConvolution");
    filterWeightsShape[0] /= numGroups;
    filterWeightsShape[1] /= numGroups;
    filterWeightsShape.insert(filterWeightsShape.begin(), numGroups);
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeGroupConvolution(in, filterWeightsNode, type, strides, padsBegin, padsEnd, dilations, autoPad, addBiases, biasesWeights);
}

std::shared_ptr<Node> makeGroupConvolution(const ngraph::Output<Node> &in,
                                           const ngraph::Output<Node> &weights,
                                           const element::Type &type,
                                           const std::vector<size_t> &strides,
                                           const std::vector<ptrdiff_t> &padsBegin,
                                           const std::vector<ptrdiff_t> &padsEnd,
                                           const std::vector<size_t> &dilations,
                                           const op::PadType &autoPad,
                                           bool addBiases,
                                           const std::vector<float> &biasesWeights) {
    auto conv = std::make_shared<opset1::GroupConvolution>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad);
    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);
        return add;
    } else {
        return conv;
    }
}

std::shared_ptr<Node> makeGroupConvolutionRelaxed(const ngraph::Output<Node> &in,
                                                  const element::Type &type,
                                                  const std::vector<size_t> &filterSize,
                                                  const std::vector<size_t> &strides,
                                                  const std::vector<ptrdiff_t> &padsBegin,
                                                  const std::vector<ptrdiff_t> &padsEnd,
                                                  const std::vector<size_t> &dilations,
                                                  const op::PadType &autoPad,
                                                  size_t numOutChannels,
                                                  size_t numGroups,
                                                  const std::vector<float> &filterWeights) {
    auto inputParamsFP32 = ngraph::builder::makeParams(ngraph::element::f32, { in.get_shape() });
    auto paramOutsFP32 = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParamsFP32));

    auto groupConvolutionNodeRelaxed = std::make_shared<op::TypeRelaxed<opset1::GroupConvolution>>(
            *as_type_ptr<opset1::GroupConvolution>(ngraph::builder::makeGroupConvolution(
                    paramOutsFP32.front(), ngraph::element::f32, filterSize, strides, padsBegin, padsEnd, dilations, autoPad, numOutChannels, numGroups)),
            element::f32);

    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, shape[1]};
    if (filterWeightsShape[0] % numGroups || filterWeightsShape[1] % numGroups)
        throw std::runtime_error("incorrected shape for GroupConvolution");
    filterWeightsShape[0] /= numGroups;
    filterWeightsShape[1] /= numGroups;
    filterWeightsShape.insert(filterWeightsShape.begin(), numGroups);
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    auto newGroupConvolution = groupConvolutionNodeRelaxed->copy_with_new_inputs({in, filterWeightsNode});

    return newGroupConvolution;
}

}  // namespace builder
}  // namespace ngraph
