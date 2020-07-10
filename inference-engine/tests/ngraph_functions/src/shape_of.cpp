// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
//
//        std::shared_ptr<ngraph::Node> makeShapeOf(std::vector<ngraph::Output<Node>> &in,
//                                                  const element::Type output_type) {
//            auto conv = std::make_shared<opset1::Convolution>(in, filterWeightsNode, strides, padsBegin, padsEnd, dilations,
//                                                              autoPad);
//            if (addBiases) {
//                bool randomBiases = biasesWeights.empty();
//                auto biasesWeightsNode = makeConstant(type, {}, biasesWeights, randomBiases);
//                auto add = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);
//                return add;
//            } else {
//                return conv;
//            }
//        }

}  // namespace builder
}  // namespace ngraph