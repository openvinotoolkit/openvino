// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef GRAPH_COMPILER

#include "intel_gpu/op/mlir_op.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/mlir_primitive.hpp"

namespace ov::op::internal {
using MLIR = ov::intel_gpu::op::MLIROp;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateMLIROp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MLIR>& op) {
    auto inputs = p.GetInputInfo(op);
    const std::string layer_name = layer_type_name_ID(op);
    const size_t num_outputs = op->get_output_size();

    cldnn::mlir_primitive::shape_infer_function shape_infer_f =
        [op](const std::vector<ov::PartialShape>& input_shapes) {
            return op->shape_infer(input_shapes);
        };

    cldnn::mlir_primitive primitive(layer_name,
                                    inputs,
                                    op,  // shared_ptr<ov::Node>
                                    std::move(shape_infer_f),
                                    num_outputs,
                                    get_output_data_types(op));

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(internal, MLIR);

}  // namespace ov::intel_gpu

#endif  // GRAPH_COMPILER
