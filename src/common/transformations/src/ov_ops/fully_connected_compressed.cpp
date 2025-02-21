// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fully_connected_compressed.hpp"

#include <memory>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace op {
namespace internal {

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& X,
                                                   const ov::Output<Node>& W,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& weight_scales,
                                                   const ov::Output<Node>& weight_zero_points,
                                                   const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, weight_scales);
    set_argument(4, weight_zero_points);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& X,
                                                   const ov::Output<Node>& W,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& weight_scales,
                                                   const ov::element::Type output_type)
    : FullyConnectedCompressed(X,
                               W,
                               bias,
                               weight_scales,
                               std::make_shared<v0::Constant>(element::dynamic, Shape{0}),
                               output_type) {}

std::shared_ptr<ov::Node> FullyConnectedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                      new_args.at(1),
                                                      new_args.at(2),
                                                      new_args.at(3),
                                                      new_args.at(4),
                                                      m_output_type);
}

// @todo finalize validate_and_infer_types
void FullyConnectedCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();

    NODE_VALIDATION_CHECK(this, input_size == 5, "Number of inputs is incorrect. Current value is: ", input_size);

    FullyConnected::validate_and_infer_types();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
