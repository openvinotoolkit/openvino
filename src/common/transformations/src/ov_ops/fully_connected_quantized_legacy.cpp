// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fully_connected_quantized_legacy.hpp"

#include <memory>

#include "matmul_shape_inference.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace op {
namespace internal {

FullyConnectedQuantizedLegacy::FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                                             const ov::Output<Node>& W,
                                                             const ov::Output<Node>& bias,
                                                             const ov::Output<Node>& deq_scales,
                                                             const ov::Output<Node>& deq_zero_points,
                                                             const ov::element::Type output_type)
    : FullyConnected(X, W, bias, output_type) {
    set_argument(3, deq_scales);
    set_argument(4, deq_zero_points);
    validate_and_infer_types();
}

FullyConnectedQuantizedLegacy::FullyConnectedQuantizedLegacy(const ov::Output<Node>& X,
                                                             const ov::Output<Node>& W,
                                                             const ov::Output<Node>& bias,
                                                             const ov::Output<Node>& deq_scales,
                                                             const ov::element::Type output_type)
    : FullyConnectedQuantizedLegacy(X,
                                    W,
                                    bias,
                                    deq_scales,
                                    std::make_shared<v0::Constant>(element::dynamic, Shape{0}),
                                    output_type) {}

std::shared_ptr<ov::Node> FullyConnectedQuantizedLegacy::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnectedQuantizedLegacy>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           m_output_type);
}

// @todo finalize validate_and_infer_types
void FullyConnectedQuantizedLegacy::validate_and_infer_types() {
    const auto input_size = get_input_size();

    NODE_VALIDATION_CHECK(this, input_size == 5, "Number of inputs is incorrect. Current value is: ", input_size);

    ov::op::v0::MatMul op;
    op.set_transpose_a(false);
    op.set_transpose_b(true);

    auto out_shapes =
        ov::op::v0::shape_infer(&op,
                                std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)});
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
