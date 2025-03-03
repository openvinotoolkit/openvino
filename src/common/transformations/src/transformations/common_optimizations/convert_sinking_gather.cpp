// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_sinking_gather.hpp"

#include <algorithm>
#include <memory>
#include <openvino/opsets/opset8.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertSinkingGather::ConvertSinkingGather() {
    MATCHER_SCOPE(ConvertSinkingGather);
    auto data = pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(ov::element::f16));
    auto convert = pattern::wrap_type<ov::op::v0::Convert>({data}, pattern::consumers_count(1));
    auto index = pattern::any_input();
    auto axis = pattern::any_input();
    auto gather = pattern::wrap_type<ov::op::v8::Gather>({convert, index, axis});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto data_input = pattern_map.at(data);
        auto idx_input = pattern_map.at(index);
        auto axis_input = pattern_map.at(axis);
        auto convert_node = pattern_map.at(convert).get_node_shared_ptr();
        auto gather_node = pattern_map.at(gather).get_node_shared_ptr();

        if (convert_node->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        if (convert_node->get_input_element_type(0) != element::Type_t::f16 ||
            convert_node->get_output_element_type(0) != element::Type_t::f32) {
            return false;
        }

        auto new_gather = gather_node->clone_with_new_inputs({data_input, idx_input, axis_input});
        auto new_convert = convert_node->clone_with_new_inputs({new_gather});
        register_new_node(new_gather);

        new_convert->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({convert_node, gather_node}, {new_gather, new_convert});
        replace_node(m.get_match_root(), new_convert);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather, matcher_name);
    this->register_matcher(m, callback);
}