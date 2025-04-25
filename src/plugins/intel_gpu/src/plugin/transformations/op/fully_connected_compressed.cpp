// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected_compressed.hpp"

namespace ov::intel_gpu::op {

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& w_decompression_scale,
                                                   const ov::Output<Node>& w_decompression_zero_point,
                                                   const ov::Output<Node>& a_decompression_scale,
                                                   const ov::Output<Node>& a_decompression_zero_point,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, bias, output_type) {
    set_argument(3, w_decompression_scale);
    set_argument(4, w_decompression_zero_point);
    set_argument(5, a_decompression_scale);
    set_argument(6, a_decompression_zero_point);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& w_decompression_scale,
                                                   const ov::Output<Node>& w_decompression_zero_point,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, bias, output_type) {
    set_argument(3, w_decompression_scale);
    set_argument(4, w_decompression_zero_point);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& bias,
                                                   const ov::Output<Node>& w_decompression_scale,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, bias, output_type) {
    set_argument(3, w_decompression_scale);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnectedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 4)
        return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          new_args.at(3),
                                                          m_output_type);
    else if (new_args.size() == 5)
        return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          new_args.at(3),
                                                          new_args.at(4),
                                                          m_output_type);
    else if (new_args.size() == 7)
        return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          new_args.at(3),
                                                          new_args.at(4),
                                                          new_args.at(5),
                                                          new_args.at(6),
                                                          m_output_type);
    else
        OPENVINO_THROW("Unexpected inputs count for FullyConnectedCompressed op: ", new_args.size());}
}  // namespace ov::intel_gpu::op
