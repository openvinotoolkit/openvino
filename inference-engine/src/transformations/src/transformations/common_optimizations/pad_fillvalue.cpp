// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pad_fillvalue.hpp"
#include "transformations/utils/utils.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/validation_util.hpp>

using namespace ngraph;
using namespace pass;

NGRAPH_RTTI_DEFINITION(PadFill, "PadFill", 0);
NGRAPH_RTTI_DEFINITION(PadFillValue, "PadFillValue", 0);

PadFillValue::PadFillValue() {
    MATCHER_SCOPE(PadFillValue);

    auto data_pattern       = pattern::any_input();
    auto pads_begin_pattern = pattern::wrap_type<opset5::Constant>();
    auto pads_end_pattern   = pattern::wrap_type<opset5::Constant>();
    auto pad_value_pattern  = pattern::wrap_type<opset5::Constant>();
    auto pad_node_pattern   = pattern::wrap_type<opset5::Pad>({data_pattern, pads_begin_pattern,
                                                            pads_end_pattern, pad_value_pattern},
                                                            pattern::consumers_count(1));
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];

        auto pad           = std::dynamic_pointer_cast<opset5::Pad>(pattern_map[pad_node_pattern].get_node_shared_ptr());
        auto pads_begin    = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_begin_pattern].get_node_shared_ptr());
        auto pads_end      = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pads_end_pattern].get_node_shared_ptr());
        auto pad_val_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map[pad_value_pattern].get_node_shared_ptr());

        if (pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT && !pad_val_const->get_input_size()) {
            auto pad_begin_value = pads_begin->get_vector<int64_t>();
            auto pad_end_value   = pads_end->get_vector<int64_t>();

            auto pads_b = opset5::Constant::create(pads_begin->get_element_type(), Shape{4}, pad_begin_value);
            auto pads_e = opset5::Constant::create(pads_end->get_element_type(),   Shape{4}, pad_end_value);
            auto val    = opset5::Constant::create(data.get_element_type(), Shape{}, {0});

            auto new_conv = std::make_shared<opset5::ConvertLike>(val, data);
            pad_val_const->set_friendly_name(pad_val_const->get_friendly_name() + "/pad_value_convert_input_port");
            new_conv->set_friendly_name(new_conv->get_friendly_name() + "/pad_value_convert");
            auto new_pad = std::make_shared<opset5::Pad>(pad->get_input_node_shared_ptr(0), pads_b, pads_e, new_conv, op::PadMode::CONSTANT);
            new_pad->set_friendly_name(pad->get_friendly_name() + "/converted");

            copy_runtime_info(pad, new_pad);
            replace_node(pad, new_pad);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(pad_node_pattern, matcher_name);
    register_matcher(m, callback);
}
