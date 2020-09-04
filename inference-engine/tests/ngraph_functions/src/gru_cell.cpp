// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeGRUCell(const OutputVector& in,
                                          const std::vector<ngraph::Shape>& WRB,
                                          std::size_t hidden_size,
                                          const std::vector<std::string>& activations,
                                          const std::vector<float>& activations_alpha,
                                          const std::vector<float>& activations_beta,
                                          float clip,
                                          bool linear_before_reset) {
    std::vector<float> empty;
    auto W = ngraph::builder::makeConstant(in[0].get_element_type(), WRB[0], empty, true);
    auto R = ngraph::builder::makeConstant(in[0].get_element_type(), WRB[1], empty, true);
    auto B = ngraph::builder::makeConstant(in[0].get_element_type(), WRB[2], empty, true);
    return std::make_shared<ngraph::opset4::GRUCell>(in[0], in[1], W, R, B, hidden_size, activations,
            activations_alpha, activations_beta, clip, linear_before_reset);
}

}  // namespace builder
}  // namespace ngraph