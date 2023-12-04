// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/group_conv.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGroupConvolution(const ov::Output<Node>& in,
                                           const element::Type& type,
                                           const std::vector<size_t>& filterSize,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& padsBegin,
                                           const std::vector<ptrdiff_t>& padsEnd,
                                           const std::vector<size_t>& dilations,
                                           const op::PadType& autoPad,
                                           size_t numOutChannels,
                                           size_t numGroups,
                                           bool addBiases,
                                           const std::vector<float>& filterWeights,
                                           const std::vector<float>& biasesWeights) {
    bool randomFilterWeights = filterWeights.empty();
    auto shape = in.get_partial_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, static_cast<size_t>(shape[1].get_length())};
    if (filterWeightsShape[0] % numGroups || filterWeightsShape[1] % numGroups)
        throw std::runtime_error("incorrected shape for GroupConvolution");
    filterWeightsShape[0] /= numGroups;
    filterWeightsShape[1] /= numGroups;
    filterWeightsShape.insert(filterWeightsShape.begin(), numGroups);
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = makeConstant(type, filterWeightsShape, filterWeights, randomFilterWeights);

    return makeGroupConvolution(in,
                                filterWeightsNode,
                                type,
                                strides,
                                padsBegin,
                                padsEnd,
                                dilations,
                                autoPad,
                                addBiases,
                                biasesWeights);
}

std::shared_ptr<Node> makeGroupConvolution(const ov::Output<Node>& in,
                                           const ov::Output<Node>& weights,
                                           const element::Type& type,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& padsBegin,
                                           const std::vector<ptrdiff_t>& padsEnd,
                                           const std::vector<size_t>& dilations,
                                           const op::PadType& autoPad,
                                           bool addBiases,
                                           const std::vector<float>& biasesWeights) {
    auto conv =
        std::make_shared<ov::op::v1::GroupConvolution>(in, weights, strides, padsBegin, padsEnd, dilations, autoPad);
    if (addBiases) {
        bool randomBiases = biasesWeights.empty();
        auto biasesWeightsNode = makeConstant(type, {}, biasesWeights, randomBiases);
        auto add = std::make_shared<ov::op::v1::Add>(conv, biasesWeightsNode);
        return add;
    } else {
        return conv;
    }
}

}  // namespace builder
}  // namespace ngraph
