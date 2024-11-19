// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fully_connected_quantized.hpp"

#include "openvino/core/type/element_type.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace op {
namespace internal {

FullyConnectedQuantized::FullyConnectedQuantized(const OutputVector& arguments, const ov::element::Type output_type)
    : FullyConnected(OutputVector(arguments.begin(), arguments.begin() + 3), output_type) {
    for (size_t i = 3; i < arguments.size(); i++) {
        set_argument(i, arguments[i]);
    }
    validate_and_infer_types();
}

FullyConnectedQuantized::FullyConnectedQuantized(const ov::Output<Node>& X,
                                                 const ov::Output<Node>& W,
                                                 const ov::Output<Node>& bias,
                                                 const ov::Output<Node>& weight_scales,
                                                 const ov::Output<Node>& weight_zero_points,
                                                 const ov::Output<Node>& input_scales,
                                                 const ov::Output<Node>& input_zero_points,
                                                 const ov::Output<Node>& output_scales,
                                                 const ov::Output<Node>& output_zero_points,
                                                 const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, weight_scales);
    set_argument(4, weight_zero_points);
    set_argument(5, input_scales);
    set_argument(6, input_zero_points);
    set_argument(7, output_scales);
    set_argument(8, output_zero_points);
    validate_and_infer_types();
}

FullyConnectedQuantized::FullyConnectedQuantized(const ov::Output<Node>& X,
                                                 const ov::Output<Node>& W,
                                                 const ov::Output<Node>& bias,
                                                 const ov::Output<Node>& weight_scales,
                                                 const ov::Output<Node>& weight_zero_points,
                                                 const ov::Output<Node>& input_scales,
                                                 const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, weight_scales);
    set_argument(4, weight_zero_points);
    set_argument(5, input_scales);
    validate_and_infer_types();
}

FullyConnectedQuantized::FullyConnectedQuantized(const ov::Output<Node>& X,
                                                 const ov::Output<Node>& W,
                                                 const ov::Output<Node>& bias,
                                                 const ov::Output<Node>& weight_scales,
                                                 const ov::Output<Node>& weight_zero_points,
                                                 const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, weight_scales);
    set_argument(4, weight_zero_points);
}

FullyConnectedQuantized::FullyConnectedQuantized(const ov::Output<Node>& X,
                                                 const ov::Output<Node>& W,
                                                 const ov::Output<Node>& bias,
                                                 const ov::Output<Node>& weight_scales,
                                                 const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, weight_scales);
}

std::shared_ptr<ov::Node> FullyConnectedQuantized::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnectedQuantized>(new_args, m_output_type);
}

std::shared_ptr<Node> FullyConnectedQuantized::fuse_bias(const ov::Output<Node>& bias) const {
    auto inputs = input_values();
    inputs[2] = bias;

    return std::make_shared<FullyConnectedQuantized>(inputs, get_output_type());
}

// @todo finalize validate_and_infer_types
void FullyConnectedQuantized::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size >= 4,
                          "Number of inputs is incorrect. Current value is: ",
                          input_size,
                          ", expected at least 3.");

    FullyConnected::validate_and_infer_types();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
