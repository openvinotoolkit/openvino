// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_squeeze(const NodeContext& context) {
    num_inputs_check(context, 1, 2, true);  // allow_complex = true
    auto x = context.get_input(0);

    auto complex = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());

    if (complex) {
        auto data = complex->get_input_source_output(0);
        std::shared_ptr<Node> res;

        if (context.input_is_none(1)) {
            res = context.mark_node(std::make_shared<v0::Squeeze>(data));
        } else {
            auto dim = context.get_input(1);
            if (dim.get_element_type() != element::i32) {
                dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
            }
            auto rank = std::get<1>(get_shape_rank(context, x, true));
            dim = normalize_axis(context, dim, rank);
            res = context.mark_node(std::make_shared<v0::Squeeze>(data, dim));
        }

        return {context.mark_node(std::make_shared<ComplexTypeMark>(res->output(0), complex->get_complex_part_type()))};
    }

    // Original non-complex path
    if (context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<v0::Squeeze>(x))};
    }
    return {context.mark_node(std::make_shared<v0::Squeeze>(x, context.get_input(1)))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
