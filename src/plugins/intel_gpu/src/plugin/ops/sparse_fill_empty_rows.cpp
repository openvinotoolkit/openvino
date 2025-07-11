// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_fill_empty_rows.hpp"

namespace ov::intel_gpu {

static void CreateSparseFillEmptyRowsOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::SparseFillEmptyRows>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::sparse_fill_empty_rows(layer_type_name_ID(op), inputs);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v16, SparseFillEmptyRows);

}  // namespace ov::intel_gpu
