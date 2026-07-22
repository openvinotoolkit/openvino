// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/segment_max.hpp"

namespace ov::intel_gpu {

static void CreateSegmentMaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::SegmentMax>& op) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);

    auto prim = cldnn::segment_max(layer_type_name_ID(op),
                                   inputs[0],   // data
                                   inputs[1],   // segment_ids
                                   op->get_fill_mode());

    // Store max(segment_ids) for compile-time shape inference.
    // Use get_constant_from_source to handle cases where the constant is
    // wrapped by Convert/Reshape/etc.
    auto seg_const = ov::util::get_constant_from_source(op->input_value(1));

    // When num_segments is not provided (2-input form), segment_ids must be a Constant
    // so that the primitive can infer the output shape based on max(segment_ids) + 1.
    if (op->get_input_size() == 2) {
        OPENVINO_ASSERT(seg_const,
                        "[GPU] SegmentMax: segment_ids input must be a Constant when num_segments "
                        "is not provided. Non-constant segment_ids in 2-input form is not supported.");
    }

    if (seg_const) {
        // segment_ids is required to be sorted (non-decreasing), so back() == max.
        auto seg_ids = seg_const->cast_vector<int64_t>();
        if (!seg_ids.empty()) {
            prim.max_segment_id = seg_ids.back();
        }
    }

    // Store num_segments constant data for compile-time shape inference.
    // Note: The current primitive/kernel only supports num_segments as a compile-time constant.
    // A non-constant num_segments tensor is not wired as a primitive input.
    if (op->get_input_size() > 2) {
        auto ns_const = ov::util::get_constant_from_source(op->input_value(2));
        OPENVINO_ASSERT(ns_const, "[GPU] SegmentMax: num_segments input must be a Constant. "
                                  "Non-constant num_segments is not yet supported.");
        prim.num_segments_val = ns_const->cast_vector<int64_t>()[0];
    }

    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v16, SegmentMax);

}  // namespace ov::intel_gpu
