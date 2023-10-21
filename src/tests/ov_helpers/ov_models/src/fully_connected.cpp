// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeFullyConnected(const ov::Output<Node>& in,
                                         const element::Type& type,
                                         const size_t outputSize,
                                         bool addBias,
                                         const ov::Shape& weightsShape,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biasWeights) {
    auto shape = weightsShape;
    if (shape.empty()) {
        auto inputShape = in.get_shape();
        shape = {inputShape[1], outputSize};
    }

    bool randomWeights = weights.empty();
    auto weightsNode = makeConstant(type, shape, weights, randomWeights);

    auto fc = std::make_shared<ov::op::v0::MatMul>(in, weightsNode, false, false);
    fc->set_friendly_name("FullyConnected");

    if (addBias) {
        bool randomBiasWeights = biasWeights.empty();
        auto biasWeightsNode = makeConstant(type, {}, biasWeights, randomBiasWeights);
        auto add = std::make_shared<ov::op::v1::Add>(fc, biasWeightsNode);

        return add;
    } else {
        return fc;
    }
}

}  // namespace builder
}  // namespace ngraph
