// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_view_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

AtenViewReplacer::AtenViewReplacer() {
    auto view_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto view_op = cast_fw_node(m.get_match_root(), "aten::view");
        if (!view_op) {
            return false;
        }
        auto shape_node = view_op->input_value(1).get_node_shared_ptr();
        if (auto list_unpack_node = cast_fw_node(shape_node, "prim::ListConstruct")) {
            OutputVector inputs;
            auto axis_0 = opset10::Constant::create(element::i64, Shape{}, {0});
            for (auto& input : shape_node->inputs()) {
                auto rank = input.get_partial_shape().rank();
                FRONT_END_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() == 0, "Rank must be 0");
                auto converted_in = std::make_shared<opset10::Convert>(input.get_source_output(), element::i64);
                auto unsqueeze = std::make_shared<opset10::Unsqueeze>(converted_in, axis_0);
                inputs.push_back(unsqueeze);
            }
            auto concat = std::make_shared<opset10::Concat>(inputs, 0);
            auto reshape = std::make_shared<opset10::Reshape>(view_op->get_input_source_output(0), concat, false);
            copy_runtime_info({view_op, shape_node}, reshape);
            replace_node(view_op, reshape);
            return true;
        };
        if (auto constant_node = std::dynamic_pointer_cast<opset10::Constant>(shape_node)) {
            // prim::Constant case
            auto reshape = std::make_shared<opset10::Reshape>(view_op->get_input_source_output(0),
                                                              view_op->get_input_source_output(1),
                                                              false);
            copy_runtime_info(view_op, reshape);
            replace_node(view_op, reshape);
            return true;
        };
        if (auto size_node = std::dynamic_pointer_cast<opset10::ShapeOf>(shape_node)) {
            // aten::shape case
            auto reshape = std::make_shared<opset10::Reshape>(view_op->get_input_source_output(0),
                                                              view_op->get_input_source_output(1),
                                                              false);
            copy_runtime_info(view_op, reshape);
            replace_node(view_op, reshape);
            return true;
        };
        if (auto slice_node = std::dynamic_pointer_cast<opset10::Slice>(shape_node)) {
            // aten::slice case
            auto reshape = std::make_shared<opset10::Reshape>(view_op->get_input_source_output(0),
                                                              view_op->get_input_source_output(1),
                                                              false);
            copy_runtime_info(view_op, reshape);
            replace_node(view_op, reshape);
            return true;
        };
        return false;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(view_op, "ov::frontend::pytorch::pass::AtenViewReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
