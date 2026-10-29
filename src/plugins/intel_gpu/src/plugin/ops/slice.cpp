// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/op_types.hpp"

#include "intel_gpu/primitives/slice.hpp"

#include <memory>

namespace ov::intel_gpu {

namespace {

static void CreateSliceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::Slice>& op) {
    validate_inputs_count(op, { 4, 5 });
    auto inputs = p.GetInputInfo(op);

    int32_t non_constant_input_mask = 0;

    std::vector<int64_t> start_constant;
    if (auto start_node = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr())) {
        start_constant = start_node->cast_vector<int64_t>();
    } else {
        non_constant_input_mask |= cldnn::slice::SLICE_NON_CONST_INPUT::START;
    }

    std::vector<int64_t> step_constant;
    if (auto step_node = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(3).get_node_shared_ptr())) {
        step_constant = step_node->cast_vector<int64_t>();
    } else {
        non_constant_input_mask |= cldnn::slice::SLICE_NON_CONST_INPUT::STEP;
    }

    std::vector<int64_t> axes_constant;
    if (op->get_input_size() == 5) {
        if (auto axes_node = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(4).get_node_shared_ptr())) {
            axes_constant = axes_node->cast_vector<int64_t>();
        } else {
            non_constant_input_mask |= cldnn::slice::SLICE_NON_CONST_INPUT::AXES;
        }
    }

    cldnn::slice slice_prim {layer_type_name_ID(op), inputs, non_constant_input_mask, start_constant, step_constant, axes_constant};
    p.add_primitive(*op, slice_prim);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, Slice);

}  // namespace ov::intel_gpu
