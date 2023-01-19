// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "min_prim_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

MinPrimListConstructReplacer::MinPrimListConstructReplacer() {
    auto min_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto min_op = cast_fw_node(m.get_match_root(), "prim::min");
        if (!min_op) {
            return false;
        }
        auto input_node = min_op->input_value(0).get_node_shared_ptr();
        auto num_inputs = min_op->inputs().size();
        auto input = concat_list_construct(input_node);
        if (num_inputs == 1) {
            auto start = std::make_shared<opset10::Constant>(element::i32, Shape{}, 0);
            auto step = std::make_shared<opset10::Constant>(element::i32, Shape{}, 1);
            auto shape = std::make_shared<opset10::ShapeOf>(input, element::i32);
            auto rank = std::make_shared<opset10::ShapeOf>(shape, element::i32);
            auto reduced_rank = std::make_shared<opset10::Squeeze>(rank);
            auto axes = std::make_shared<opset10::Range>(start, reduced_rank, step, element::i32);
            auto reduce_min = std::make_shared<opset10::ReduceMin>(input, axes);
            copy_runtime_info({min_op, input_node}, reduce_min);
            replace_node(min_op, reduce_min);
            return true;
        }
        auto second_input_node = min_op->input_value(1).get_node_shared_ptr();
        auto second_input = concat_list_construct(second_input_node);
        auto minimum_op = std::make_shared<opset10::Minimum>(input, second_input);
        copy_runtime_info({min_op, input_node, second_input_node}, minimum_op);
        replace_node(min_op, minimum_op);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(min_op,
                                                          "ov::frontend::pytorch::pass::MinPrimListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov