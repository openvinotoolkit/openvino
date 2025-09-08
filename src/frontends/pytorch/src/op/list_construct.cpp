// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_construct(const NodeContext& context) {
    // Process the case when prim::ListConstruct has all inputs constant
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    ov::OutputVector consts;
    for (size_t i = 0; i < context.get_input_size(); i++) {
        auto input = context.get_input_from_visible_context(i);
        auto c_node = ov::as_type_ptr<v0::Constant>(input.get_node_shared_ptr());
        PYTORCH_OP_CONVERSION_CHECK(c_node, "Translation for prim::ListConstruct support only constant inputs");
        if (c_node->get_shape().size() == 0) {
            c_node = std::make_shared<v0::Constant>(c_node->get_element_type(), Shape{1}, c_node->get_data_ptr());
            consts.push_back(c_node);
        } else {
            auto unsqueezed_c_node = context.mark_node(std::make_shared<v0::Unsqueeze>(c_node, const_0));
            consts.push_back(unsqueezed_c_node);
        }
    }
    auto list_construct = context.mark_node(std::make_shared<v0::Concat>(consts, 0));
    if (list_construct->has_evaluate()) {
        OutputVector replacements(list_construct->get_output_size());

        if (list_construct->constant_fold(replacements, list_construct->input_values())) {
            return replacements;
        }
    }
    return {list_construct};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
