// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/segment_max.hpp"

namespace ov::intel_gpu {

static void CreateSegmentMaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::SegmentMax>& op) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);

    int fill_mode = (op->get_fill_mode() == ov::op::FillMode::ZERO) ? 0 : 1;

    auto prim = cldnn::segment_max(layer_type_name_ID(op),
                                   inputs[0],   // data
                                   inputs[1],   // segment_ids
                                   fill_mode);

    // Store segment_ids constant data for compile-time shape inference
    if (auto seg_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            op->input_value(1).get_node_shared_ptr())) {
        prim.segment_ids_data = seg_const->cast_vector<int64_t>();
    }

    // Store num_segments constant data for compile-time shape inference
    if (op->get_input_size() > 2) {
        if (auto ns_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                op->input_value(2).get_node_shared_ptr())) {
            prim.num_segments_val = ns_const->cast_vector<int64_t>()[0];
        }
    }

    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v16, SegmentMax);

}  // namespace ov::intel_gpu
