// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/slice.hpp"

#include "intel_gpu/primitives/slice.hpp"

#include <memory>

namespace ov::intel_gpu {

namespace {

static void CreateSliceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::Slice>& op) {
    validate_inputs_count(op, { 4, 5 });
    auto inputs = p.GetInputInfo(op);
    cldnn::slice slice_prim {layer_type_name_ID(op), inputs};
    p.add_primitive(*op, slice_prim);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, Slice);

}  // namespace ov::intel_gpu
