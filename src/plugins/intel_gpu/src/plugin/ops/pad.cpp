// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/pad.hpp"

#include "intel_gpu/primitives/border.hpp"

namespace ov {
namespace intel_gpu {

static void CreatePadOp(Program& p, const std::shared_ptr<ngraph::op::v1::Pad>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::input_info> non_constant_inputs = {inputs[0]};
    int32_t non_constant_input_mask = 0;

    auto pads_begin_constant = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
    std::vector<int64_t> pads_begin = std::vector<int64_t>{};
    if (pads_begin_constant) {
        pads_begin = pads_begin_constant->cast_vector<int64_t>();
    } else {
        non_constant_inputs.push_back(inputs[1]);
        non_constant_input_mask |= cldnn::border::PAD_NON_CONST_INPUT::BEGIN;
    }

    auto pads_end_constant = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(2).get_node_shared_ptr());
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
        auto const_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        if (const_node) {
            ov::op::util::get_single_value(const_node, pad_value);
            is_value_const = true;
        }
    }

    if (!is_value_const) {
        non_constant_inputs.push_back(inputs[3]);
        non_constant_input_mask |= cldnn::border::PAD_NON_CONST_INPUT::VALUE;
    }

    auto tilePrim = cldnn::border(layerName,
                                  non_constant_inputs,
                                  non_constant_input_mask,
                                  pads_begin,
                                  pads_end,
                                  op->get_pad_mode(),
                                  pad_value);

    p.add_primitive(*op, tilePrim);
}

REGISTER_FACTORY_IMPL(v1, Pad);

}  // namespace intel_gpu
}  // namespace ov
