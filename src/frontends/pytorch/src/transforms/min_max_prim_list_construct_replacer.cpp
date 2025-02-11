// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "min_max_prim_list_construct_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

MinMaxPrimListConstructReplacer::MinMaxPrimListConstructReplacer() {
    const auto& op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        bool is_min = false;
        const auto& max_op = cast_fw_node(m.get_match_root(), "prim::max");
        const auto& min_op = cast_fw_node(m.get_match_root(), "prim::min");
        std::shared_ptr<ov::op::util::FrameworkNode> op;
        if (!max_op && !min_op) {
            return false;
        }
        if (min_op != nullptr) {
            is_min = true;
            op = min_op;
        } else {
            op = max_op;
        }
        ov::pass::NodeRegistry rg;
        auto input_node = op->input_value(0);
        auto num_inputs = op->inputs().size();
        auto input = concat_list_construct(input_node);
        std::shared_ptr<Node> reduce_op;
        if (num_inputs == 1) {
            auto start = rg.make<v0::Constant>(element::i32, Shape{}, 0);
            auto step = rg.make<v0::Constant>(element::i32, Shape{}, 1);
            auto shape = rg.make<v3::ShapeOf>(input, element::i32);
            auto rank = rg.make<v3::ShapeOf>(shape, element::i32);
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto reduced_rank = rg.make<v0::Squeeze>(rank, axis_0);
            auto axes = rg.make<v4::Range>(start, reduced_rank, step, element::i32);
            std::shared_ptr<Node> reduce_op;
            if (!is_min) {
                reduce_op = rg.make<v1::ReduceMax>(input, axes);
            } else {
                reduce_op = rg.make<v1::ReduceMin>(input, axes);
            }
            copy_runtime_info_and_name(op, rg.get());
            replace_node(op, reduce_op);
            return true;
        }
        auto second_input_node = op->input_value(1);
        auto second_input = concat_list_construct(second_input_node);
        std::shared_ptr<Node> min_or_max_op;
        if (!is_min) {
            min_or_max_op = rg.make<v1::Maximum>(input, second_input);
        } else {
            min_or_max_op = rg.make<v1::Minimum>(input, second_input);
        }
        copy_runtime_info_and_name(op, rg.get());
        replace_node(op, min_or_max_op);
        return true;
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(op,
                                                     "ov::frontend::pytorch::pass::MinMaxPrimListConstructReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
