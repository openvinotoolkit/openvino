// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/cast_to_convert.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass::pattern;

ov::pass::CastToConvert::CastToConvert() {
    MATCHER_SCOPE(CastToConvert);

    auto param_pattern = pattern::wrap_type<v0::Parameter>();
    auto first_convert = wrap_type<v0::Convert>({param_pattern});
    auto second_convert = optional<v0::Convert>({first_convert});
    auto result = wrap_type<v0::Result>({second_convert});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& first_convert_node = pattern_map.at(first_convert).get_node_shared_ptr();
        auto first_convert_op = ov::as_type_ptr<v0::Convert>(first_convert_node);
        if (!first_convert_op) {
            return false;
        }

        auto input = first_convert_op->input_value(0);
        auto first_dest_type = first_convert_op->get_destination_type();
        auto new_first_convert = std::make_shared<v0::Convert>(input, first_dest_type, true, true);
        new_first_convert->set_friendly_name(first_convert_op->get_friendly_name());

        if (pattern_map.count(second_convert) > 0) {
            const auto& second_convert_node = pattern_map.at(second_convert).get_node_shared_ptr();
            auto second_convert_op = ov::as_type_ptr<v0::Convert>(second_convert_node);
            if (second_convert_op) {
                auto second_dest_type = second_convert_op->get_destination_type();
                auto new_second_convert =
                    std::make_shared<v0::Convert>(new_first_convert, second_dest_type, true, true);
                new_second_convert->set_friendly_name(second_convert_op->get_friendly_name());

                copy_runtime_info(first_convert_op, new_first_convert);
                copy_runtime_info(second_convert_op, new_second_convert);

                replace_node(first_convert_op, new_first_convert);
                replace_node(second_convert_op, new_second_convert);
                return true;
            }
        }

        copy_runtime_info(first_convert_op, new_first_convert);
        replace_node(first_convert_op, new_first_convert);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
