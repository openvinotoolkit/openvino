// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "listconstruct_reshape_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

ListConstructReshapeReplacer::ListConstructReshapeReplacer() {
    // Transformation for torch operators aten::view and aten::reshape for cases where second input is
    // prim::ListConstruct.
    auto reshape_op = ov::pass::pattern::wrap_type<opset10::Reshape>();
    // Both aten::view and aten::reshape are using same translation returning opset10::Reshape operator.
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto reshape_op = std::dynamic_pointer_cast<opset10::Reshape>(m.get_match_root());
        if (!reshape_op) {
            return false;
        }
        auto shape_node = reshape_op->input_value(1).get_node_shared_ptr();
        if (auto list_unpack_node = cast_fw_node(shape_node, "prim::ListConstruct")) {
            // Check if second input to operator is prim::ListConstruct, and if so, concatenate it inputs into single
            // Tensor. Concatenation is possible because all elements in list should be scalar intigers.
            OutputVector inputs;
            auto axis_0 = opset10::Constant::create(element::i64, Shape{}, {0});
            for (auto& input : shape_node->inputs()) {
                auto rank = input.get_partial_shape().rank();
                FRONT_END_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() == 0, "Rank must be 0");
                auto unsqueeze = std::make_shared<opset10::Unsqueeze>(input.get_source_output(), axis_0);
                inputs.push_back(unsqueeze);
            }
            auto concat = std::make_shared<opset10::Concat>(inputs, 0);
            copy_runtime_info({shape_node}, concat);
            replace_node(shape_node, concat);
            return true;
        };
        return false;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_op,
                                                          "ov::frontend::pytorch::pass::ListConstructReshapeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
