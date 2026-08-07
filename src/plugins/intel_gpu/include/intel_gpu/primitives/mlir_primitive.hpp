// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "primitive.hpp"

namespace ov {
class Node;  // forward-decl — the underlying op is ov::intel_gpu::op::MLIROp
}

namespace cldnn {

/// @brief Primitive that wraps an ov::intel_gpu::op::MLIROp node. Its execute_impl
/// (see impls/common/mlir_primitive.cpp) forwards to MLIROp::evaluate().
struct mlir_primitive : public primitive_base<mlir_primitive> {
    CLDNN_DECLARE_PRIMITIVE(mlir_primitive)

    using shape_infer_function =
        std::function<std::vector<ov::PartialShape>(const std::vector<ov::PartialShape>&)>;

    mlir_primitive() : primitive_base("", {}) {}

    mlir_primitive(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   std::shared_ptr<ov::Node> op,
                   shape_infer_function shape_infer_f,
                   size_t num_outputs,
                   const std::vector<optional_data_type>& out_types)
        : primitive_base(id, inputs, num_outputs, out_types),
          op(std::move(op)),
          shape_infer_f(std::move(shape_infer_f)) {}

    std::shared_ptr<ov::Node> op;
    shape_infer_function shape_infer_f;
};

}  // namespace cldnn
