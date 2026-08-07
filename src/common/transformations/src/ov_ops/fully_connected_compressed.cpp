// Copyright (C) 2018-2026 Intel Corporation
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

void FullyConnectedCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();

    NODE_VALIDATION_CHECK(this,
                          input_size == 5,
                          "FullyConnectedCompressed expects 5 inputs (X, W, bias, weight_scales, "
                          "weight_zero_points). Got: ",
                          input_size);

    // weight_scales is a mandatory input, so it must carry a concrete floating-point element type.
    // Weight zero-points are usually integral quantization offsets, but compression schemes may also
    // carry a real (e.g. f32) zero-point that is subtracted before scaling, so any numeric type is
    // accepted; an absent zero-points input is passed as an empty (element::dynamic) constant, which
    // is why a dynamic element type is accepted there.
    const auto& scales_et = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          scales_et.is_real(),
                          "weight_scales (input 3) must have a floating-point element type. Got: ",
                          scales_et);

    const auto& zp_et = get_input_element_type(4);
    NODE_VALIDATION_CHECK(this,
                          zp_et.is_real() || zp_et.is_integral_number() || zp_et.is_dynamic(),
                          "weight_zero_points (input 4) must have a numeric element type. Got: ",
                          zp_et);

    FullyConnected::validate_and_infer_types();
}

}  // namespace internal
}  // namespace op
}  // namespace ov
