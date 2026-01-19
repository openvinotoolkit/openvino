// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_gather_matmul_compressed.hpp"

#include <memory>

#include "batch_gather_matmul.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "transformations/itt.hpp"

namespace ov::intel_cpu {

BatchGatherMatmulCompressed::BatchGatherMatmulCompressed(const ov::Output<Node>& A,
                                                         const ov::Output<Node>& B,
                                                         const ov::Output<Node>& indices,
                                                         const ov::Output<Node>& bias,
                                                         const ov::Output<Node>& weight_scales,
                                                         const ov::Output<Node>& weight_zero_points)
    : BatchGatherMatmul(A, B, indices, bias) {
    set_argument(4, weight_scales);
    set_argument(5, weight_zero_points);
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> BatchGatherMatmulCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<BatchGatherMatmulCompressed>(new_args.at(0),
                                                         new_args.at(1),
                                                         new_args.at(2),
                                                         new_args.at(3),
                                                         new_args.at(4),
                                                         new_args.at(5));
}

void BatchGatherMatmulCompressed::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this, input_size == 6, "Number of inputs is incorrect. Current value is: ", input_size);

    // check weight_scales and weight_zero_points are Const
    auto weight_scales = get_input_node_shared_ptr(4);
    NODE_VALIDATION_CHECK(this,
                          ov::is_type<ov::op::v0::Constant>(weight_scales),
                          "Input weight_scales must be a Constant node.");
    auto weight_zero_points = get_input_node_shared_ptr(5);
    NODE_VALIDATION_CHECK(this,
                          ov::is_type<ov::op::v0::Constant>(weight_zero_points),
                          "Input weight_zero_points must be a Constant node.");
    BatchGatherMatmul::validate_and_infer_types();
}
}  // namespace ov::intel_cpu