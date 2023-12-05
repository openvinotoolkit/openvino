// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected_compressed.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& decompression_scale,
                                                   const ov::Output<Node>& decompression_zero_point,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, output_type) {
    set_argument(2, decompression_scale);
    set_argument(3, decompression_zero_point);
    validate_and_infer_types();
}

FullyConnectedCompressed::FullyConnectedCompressed(const ov::Output<Node>& A,
                                                   const ov::Output<Node>& B,
                                                   const ov::Output<Node>& decompression_scale,
                                                   const ov::element::Type output_type)
    : FullyConnected(A, B, output_type) {
    set_argument(2, decompression_scale);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnectedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    if (new_args.size() == 3)
        return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          m_output_type);
    else if (new_args.size() == 4)
        return std::make_shared<FullyConnectedCompressed>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          new_args.at(3),
                                                          m_output_type);
    else
        OPENVINO_THROW("Unexpected inputs count for FullyConnectedCompressed op: ", new_args.size());
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
