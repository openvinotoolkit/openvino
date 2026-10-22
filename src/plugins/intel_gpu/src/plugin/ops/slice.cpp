// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/slice.hpp"
#include "openvino/op/util/op_types.hpp"

#include "intel_gpu/primitives/slice.hpp"

#include <memory>

namespace ov::intel_gpu {

namespace {

static void CreateSliceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::Slice>& op) {
    validate_inputs_count(op, { 4, 5 });
    auto inputs = p.GetInputInfo(op);
    const bool is_start_constant = ov::op::util::is_constant(op->input_value(1).get_node());
    const bool is_step_constant = ov::op::util::is_constant(op->input_value(3).get_node());
    const bool is_axes_constant = op->get_input_size() == 5 && ov::op::util::is_constant(op->input_value(4).get_node());
    cldnn::slice slice_prim {layer_type_name_ID(op), inputs, is_start_constant, is_step_constant, is_axes_constant};
    p.add_primitive(*op, slice_prim);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, Slice);

}  // namespace ov::intel_gpu
