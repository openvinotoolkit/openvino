// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fully_connected_quantized.hpp"

#include "openvino/core/type/element_type.hpp"
#include "ov_ops/fully_connected.hpp"

namespace ov {
namespace op {
namespace internal {

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

std::shared_ptr<ov::Node> FullyConnectedQuantized::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnectedQuantized>(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     new_args.at(3),
                                                     new_args.at(4),
                                                     new_args.at(5),
                                                     new_args.at(6),
                                                     new_args.at(7),
                                                     new_args.at(8),
                                                     m_output_type);
}

void FullyConnectedQuantized::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size == 9,
                          "FullyConnectedQuantized expects 9 inputs (X, W, bias, weight_scales, "
                          "weight_zero_points, input_scales, input_zero_points, output_scales, "
                          "output_zero_points). Got: ",
                          input_size);

    // Scales are floating-point; zero-points are integral quantization offsets. An absent input is
    // passed as an empty (element::dynamic) constant, so a dynamic element type is always accepted;
    // types may also be unresolved during partial propagation.
    const auto check_scales = [this](size_t idx, const char* name) {
        const auto& et = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              et.is_real() || et.is_dynamic(),
                              name,
                              " (input ",
                              idx,
                              ") must have a floating-point element type. Got: ",
                              et);
    };
    const auto check_zero_points = [this](size_t idx, const char* name) {
        const auto& et = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              et.is_integral_number() || et.is_dynamic(),
                              name,
                              " (input ",
                              idx,
                              ") must have an integral element type. Got: ",
                              et);
    };

    check_scales(3, "weight_scales");
    check_zero_points(4, "weight_zero_points");
    check_scales(5, "input_scales");
    check_zero_points(6, "input_zero_points");
    check_scales(7, "output_scales");
    check_zero_points(8, "output_zero_points");

    FullyConnected::validate_and_infer_types();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
