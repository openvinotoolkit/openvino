// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_prim_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

MaxPrimListConstructReplacer::MaxPrimListConstructReplacer() {
    auto max_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto max_op = cast_fw_node(m.get_match_root(), "prim::max");
        if (!max_op) {
            return false;
        }
        auto input_node = max_op->input_value(0).get_node_shared_ptr();
        auto num_inputs = max_op->inputs().size();
        auto input = concat_list_construct(input_node);
        if (num_inputs == 1) {
            auto start = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
            auto step = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
            auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, element::i32);
            auto rank = std::make_shared<ov::op::v3::ShapeOf>(shape, element::i32);
            auto reduced_rank = std::make_shared<ov::op::v0::Squeeze>(rank);
            auto axes = std::make_shared<ov::op::v4::Range>(start, reduced_rank, step, element::i32);
            auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(input, axes);
            copy_runtime_info({max_op, input_node}, reduce_max);
            replace_node(max_op, reduce_max);
            return true;
        }
        auto second_input_node = max_op->input_value(1).get_node_shared_ptr();
        auto second_input = concat_list_construct(second_input_node);
        auto maximum_op = std::make_shared<ov::op::v1::Maximum>(input, second_input);
        copy_runtime_info({max_op, input_node, second_input_node}, maximum_op);
        replace_node(max_op, maximum_op);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(max_op,
                                                          "ov::frontend::pytorch::pass::MaxPrimListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov