// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/grouped_matmul.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/op/grouped_matmul.hpp"

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateGroupedMatMulOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::GroupedMatMul>& op) {
    auto inputs = p.GetInputInfo(op);
    validate_inputs_count(op, {2, 3});

    // Case 1: 3D×3D (no offsets): A:[G,M,K], B:[G,N,K]
    // Case 2: 2D×3D (with offsets): A:[T,K], B:[G,N,K], offsets:[G]
    const bool has_offsets = (op->get_input_size() == 3);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::grouped_matmul prim(layerName, inputs, has_offsets, /*has_bias=*/false);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v17, GroupedMatMul);

}  // namespace ov::intel_gpu
