// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeRNN(const OutputVector& in,
                                      const std::vector<ngraph::Shape>& constants,
                                      std::size_t hidden_size,
                                      const std::vector<std::string>& activations,
                                      const std::vector<float>& activations_alpha,
                                      const std::vector<float>& activations_beta,
                                      float clip,
                                      bool make_sequence,
                                      ngraph::op::RecurrentSequenceDirection direction) {
    std::vector<float> empty;
    auto W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true);
    auto R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true);
    auto B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true);
    if (!make_sequence) {
        return std::make_shared<ngraph::opset4::RNNCell>(in[0], in[1], W, R, B, hidden_size, activations,
                                                         activations_alpha, activations_beta, clip);
    } else {
        std::vector<float> lenghts(in[0].get_shape()[0], in[0].get_shape()[1]);
        auto seq_lenghts = ngraph::builder::makeConstant(in[0].get_element_type(), constants[3], lenghts, false);
        return std::make_shared<ngraph::op::v5::RNNSequence>(in[0], in[1], seq_lenghts, W, R, B, hidden_size, direction,
                                                             activations, activations_alpha, activations_beta, clip);
    }
}
}  // namespace builder
}  // namespace ngraph