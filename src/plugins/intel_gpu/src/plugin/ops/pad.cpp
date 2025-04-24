// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/pad.hpp"

#include "intel_gpu/primitives/border.hpp"

namespace ov::intel_gpu {

static void CreatePadOpInternal(ProgramBuilder& p, const std::shared_ptr<op::util::PadBase>& op, bool allow_negative_pad) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::input_info> non_constant_inputs = {inputs[0]};
    int32_t non_constant_input_mask = 0;

    auto pads_begin_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
    std::vector<int64_t> pads_begin = std::vector<int64_t>{};
    if (pads_begin_constant) {
        pads_begin = pads_begin_constant->cast_vector<int64_t>();
    } else {
        non_constant_inputs.push_back(inputs[1]);
        non_constant_input_mask |= cldnn::border::PAD_NON_CONST_INPUT::BEGIN;
    }

    auto pads_end_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(2).get_node_shared_ptr());
    std::vector<int64_t> pads_end = std::vector<int64_t>{};
    if (pads_end_constant) {
        pads_end = pads_end_constant->cast_vector<int64_t>();
    } else {
        non_constant_inputs.push_back(inputs[2]);
        non_constant_input_mask |= cldnn::border::PAD_NON_CONST_INPUT::END;
    }

    float pad_value = 0.f;
    bool is_value_const = false;
    if (op->get_pad_mode() == ov::op::PadMode::CONSTANT && op->get_input_size() == 4) {
        auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        if (const_node) {
            const bool check_value_range = false;  // Allows the usage of infinity value as pad_value
            OPENVINO_ASSERT(ov::op::util::get_single_value(const_node, pad_value, check_value_range),
                            "Invalid parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            is_value_const = true;
        }
    }

    if (!is_value_const) {
        non_constant_inputs.push_back(inputs[3]);
        non_constant_input_mask |= cldnn::border::PAD_NON_CONST_INPUT::VALUE;
    }

    const auto borderPrim = cldnn::border(layerName,
                                  non_constant_inputs,
                                  non_constant_input_mask,
                                  pads_begin,
                                  pads_end,
                                  op->get_pad_mode(),
                                  pad_value,
                                  allow_negative_pad);
    p.add_primitive(*op, borderPrim);
}

static void CreatePadOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Pad>& op) {
    CreatePadOpInternal(p, op, false);
}

static void CreatePadOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v12::Pad>& op) {
    CreatePadOpInternal(p, op, true);
}

REGISTER_FACTORY_IMPL(v1, Pad);
REGISTER_FACTORY_IMPL(v12, Pad);

}  // namespace ov::intel_gpu
