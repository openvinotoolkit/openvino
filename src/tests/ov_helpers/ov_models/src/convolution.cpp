// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include <memory>
#include <vector>

#include "openvino/op/add.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeConvolution(const ov::Output<Node>& in,
                                      const element::Type& type,
                                      const std::vector<size_t>& filterSize,
                                      const std::vector<size_t>& strides,
                                      const std::vector<ptrdiff_t>& padsBegin,
                                      const std::vector<ptrdiff_t>& padsEnd,
                                      const std::vector<size_t>& dilations,
                                      const op::PadType& autoPad,
                                      size_t numOutChannels,
                                      bool addBiases,
                                      const std::vector<float>& filterWeights,
                                      const std::vector<float>& biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, static_cast<size_t>(shape[1].get_length())};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);
    auto conv = std::make_shared<ov::op::v1::Convolution>(in,
                                                          filterWeightsNode,
                                                          strides,
                                                          padsBegin,
                                                          padsEnd,
                                                          dilations,
                                                          autoPad);
    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {1, numOutChannels, 1, 1}, biasesWeights, randomBiases);
        auto add = std::make_shared<ov::op::v1::Add>(conv, biasesWeightsNode);
        return add;
    } else {
        return conv;
    }
}

std::shared_ptr<Node> makeConvolution(const ov::Output<Node>& in_data,
                                      const ov::Output<Node>& in_weights,
                                      const element::Type& type,
                                      const std::vector<size_t>& filterSize,
                                      const std::vector<size_t>& strides,
                                      const std::vector<ptrdiff_t>& padsBegin,
                                      const std::vector<ptrdiff_t>& padsEnd,
                                      const std::vector<size_t>& dilations,
                                      const op::PadType& autoPad,
                                      size_t numOutChannels,
                                      bool addBiases,
                                      const std::vector<float>& biasesWeights) {
    auto shape = in_data.get_partial_shape();
    auto conv =
        std::make_shared<ov::op::v1::Convolution>(in_data, in_weights, strides, padsBegin, padsEnd, dilations, autoPad);
    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {1, numOutChannels, 1, 1}, biasesWeights, randomBiases);
        auto add = std::make_shared<ov::op::v1::Add>(conv, biasesWeightsNode);
        return add;
    } else {
        return conv;
    }
}

}  // namespace builder
}  // namespace ngraph
