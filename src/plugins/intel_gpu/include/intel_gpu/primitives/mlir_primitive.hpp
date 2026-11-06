// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "primitive.hpp"

namespace ov::intel_gpu::op {
class MLIROp;
}  // namespace ov::intel_gpu::op

namespace cldnn {

/// @brief Primitive that wraps an ov::intel_gpu::op::MLIROp node. Its execute_impl
/// (see impls/common/mlir_primitive.cpp) calls MLIRGpuProgram::execute().
struct mlir_primitive : public primitive_base<mlir_primitive> {
    CLDNN_DECLARE_PRIMITIVE(mlir_primitive)

    using shape_infer_function = std::function<std::vector<ov::PartialShape>(const std::vector<ov::PartialShape>&)>;

    mlir_primitive() : primitive_base("", {}) {}

    mlir_primitive(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   std::shared_ptr<ov::intel_gpu::op::MLIROp> op,
                   shape_infer_function shape_infer_f,
                   size_t num_outputs,
                   const std::vector<optional_data_type>& out_types)
        : primitive_base(id, inputs, num_outputs, out_types),
          op(std::move(op)),
          shape_infer_f(std::move(shape_infer_f)) {}

    std::shared_ptr<ov::intel_gpu::op::MLIROp> op;
    shape_infer_function shape_infer_f;

    // Required physical format per input (vector-index == dependency-index == MLIR function argument index).
    // The compiled MLIR program bakes the input memref layouts in, so cldnn must deliver exactly these
    // formats. If the format is format::any, the default format for the input rank is used.
    std::vector<format::type> required_input_fmts;

    [[nodiscard]] format::type get_required_input_format(size_t idx, size_t rank) const {
        if (idx < required_input_fmts.size() && required_input_fmts[idx] != format::any) {
            return required_input_fmts[idx];
        }
        return format::get_default_format(rank);
    }

    void save(BinaryOutputBuffer& /*ob*/) const override {
        OPENVINO_THROW("[GPU] Compiled model export / model caching is not supported for models with MLIR "
                       "(Graph Compiler) subgraphs. Disable GPU_ENABLE_MLIR to use ov::cache_dir or export_model()");
    }

    void load(BinaryInputBuffer& /*ib*/) override {
        OPENVINO_THROW("[GPU] Import of a compiled model containing MLIR (Graph Compiler) subgraphs "
                       "is not supported");
    }
};

}  // namespace cldnn
