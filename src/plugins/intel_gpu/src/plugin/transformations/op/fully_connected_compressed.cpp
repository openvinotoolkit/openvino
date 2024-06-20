// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected_compressed.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& decompression_scale,
                                                   const ov::Output<Node>& decompression_zero_point,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, bias, output_type)
    , m_has_zp(false)
    , m_has_activation_scale(false) {
    set_argument(3, decompression_scale);
    set_argument(4, decompression_zero_point);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& decompression_scale,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, bias, output_type)
    , m_has_zp(false)
    , m_has_activation_scale(false) {
    set_argument(3, decompression_scale);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const OutputVector& inputs,
                             bool has_zp,
                             bool has_activation_scale,
                             const ov::element::Type output_type)
    : FullyConnected(inputs[0], inputs[1], inputs[2], output_type)
    , m_has_zp(has_zp)
    , m_has_activation_scale(has_activation_scale)
{
    for (size_t i = 3; i < inputs.size(); i++)
        set_argument(i, inputs[i]);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnectedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    auto input_size = new_args.size();
    auto expected_inputs = 4;
    if (m_has_zp)
        expected_inputs++;
    if (m_has_activation_scale)
        expected_inputs++;
    NODE_VALIDATION_CHECK(this,
        input_size == m_has_zp,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected ",
        expected_inputs);

    return std::make_shared<FullyConnectedCompressed>(new_args,
                                                      m_has_zp,
                                                      m_has_activation_scale,
                                                      m_output_type);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
