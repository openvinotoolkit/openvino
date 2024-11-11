// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "check_dequantization_subgraph.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

CheckDequantizationSubgraph::CheckDequantizationSubgraph(const element::TypeVector& precisions) {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;
    // Dequantization subgraph may have two forms: with and without Subtract
    //
    //    Input                                 Input
    //      |                                     |
    //   Convert  zero point           OR       Convert   scale
    //       \     /                               \      /
    //       Subtract   scale                      Multiply
    //           \      /
    //           Multiply
    //
    auto input_pattern = any_input();
    auto convert_pattern = wrap_type<ov::op::v0::Convert>({input_pattern}, consumers_count(1));
    auto zero_point_pattern = any_input();
    auto subtract_pattern = wrap_type<ov::op::v1::Subtract>({convert_pattern, zero_point_pattern});
    auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({subtract_pattern, any_input()});
    auto multiply_no_subtract_pattern = wrap_type<ov::op::v1::Multiply>({convert_pattern, any_input()});
    auto root = std::make_shared<Or>(OutputVector{multiply_pattern, multiply_no_subtract_pattern});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert_pattern).get_node_shared_ptr();
        auto input = pattern_map.at(input_pattern);
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        auto subtract_it = pattern_map.find(subtract_pattern);
        if (subtract_it == pattern_map.end()) {
            for (size_t i = 0; i < multiply->get_input_size(); i++) {
                const auto node = ov::as_type_ptr<ov::op::v0::Convert>(multiply->get_input_node_shared_ptr(i));
                if (node && std::find(precisions.begin(), precisions.end(), node->get_input_element_type(0)) !=
                                precisions.end()) {
                    convert = node;
                    input = convert->input_value(0);
                }
            }
        }

        const auto& input_precision = input.get_element_type();
        // validation by Convert operation input precisions
        if (std::find(precisions.begin(), precisions.end(), input_precision) == precisions.end()) {
            return false;
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(root, "CheckDequantizationSubgraph");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
