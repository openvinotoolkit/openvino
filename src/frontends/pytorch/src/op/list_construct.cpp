// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_construct(NodeContext& context) {
    // Process the case when prim::ListConstruct has all inputs constant
    ov::OutputVector consts;
    for (size_t i = 0; i < context.get_input_size(); i++) {
        auto input = context.get_input_from_visible_context(i);
        auto c_node = std::dynamic_pointer_cast<v0::Constant>(input.get_node_shared_ptr());
        FRONT_END_OP_CONVERSION_CHECK(c_node, "Translation for prim::ListConstruct support only constant inputs");
        if (c_node->get_shape().size() == 0) {
            c_node = std::make_shared<v0::Constant>(c_node->get_element_type(), Shape{1}, c_node->get_data_ptr());
        }
        consts.push_back(c_node);
    }
    auto list_construct = std::make_shared<v0::Concat>(consts, 0);
    if (list_construct->has_evaluate()) {
        OutputVector replacements(list_construct->get_output_size());

        if (list_construct->constant_fold(replacements, list_construct->input_values())) {
            return replacements;
        }
    }
    return {context.mark_output(list_construct)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov