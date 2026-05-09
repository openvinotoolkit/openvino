// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/slice_scatter.hpp"

#include "intel_gpu/primitives/slice_scatter.hpp"

#include <memory>

namespace ov::intel_gpu {

namespace {

static void CreateSliceScatterOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::SliceScatter>& op) {
    validate_inputs_count(op, {5, 6});
    auto inputs = p.GetInputInfo(op);
    cldnn::slice_scatter slice_scatter_prim{layer_type_name_ID(op), inputs};
    p.add_primitive(*op, slice_scatter_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v15, SliceScatter);

}  // namespace ov::intel_gpu
