// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_stack_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace ov::pass::pattern;

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

AtenStackListConstructReplacer::AtenStackListConstructReplacer() {
    auto list_construct = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();
    auto axis = ov::pass::pattern::wrap_type<opset10::Constant>();

    // We search for a pattern: ListConstruct -> aten::stack <- Constant
    auto stack = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>({list_construct, axis});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto stack = cast_fw_node(m.get_match_root(), "aten::stack");
        if (!stack) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto input_node = pattern_map.at(list_construct).get_node_shared_ptr();
        auto axis_node = pattern_map.at(axis).get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<opset10::Constant>(axis_node);
        auto axis = axis_const->cast_vector<int64_t>();
        // Check if ListConstruct is an input
        if (auto list_construct_node = cast_fw_node(input_node, "prim::ListConstruct")) {
            const auto& list_inputs = list_construct_node->input_values();
            OutputVector node_vector;
            auto zero = opset10::Constant::create(element::i32, Shape{}, {0});
            // Iterate over values in ListConstruct
            for (const auto& list_input : list_inputs) {
                auto node = concat_list_construct(list_input.get_node_shared_ptr());
                auto unsqueezed_node = std::make_shared<opset10::Unsqueeze>(node, axis_const);
                node_vector.push_back(unsqueezed_node);
            }
            // Concat vectors on provided axis
            auto concat = std::make_shared<opset10::Concat>(node_vector, axis[0]);

            copy_runtime_info({stack, input_node}, concat);
            replace_node(stack, concat);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<Matcher>(stack, "ov::frontend::pytorch::pass::AtenStackListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
