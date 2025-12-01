// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_select(const NodeContext& context) {
    // aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
    num_inputs_check(context, 3, 3, true);  // allow_complex = true
    auto data = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.get_input(2);

    auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());

    if (complex) {
        auto underlying_data = complex->get_input_source_output(0);

        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, data, true));
        dim = normalize_axis(context, dim, rank);

        auto result = context.mark_node(std::make_shared<v8::Gather>(underlying_data, index, dim));
        return {context.mark_node(std::make_shared<ComplexTypeMark>(result, complex->get_complex_part_type()))};
    }

    return {context.mark_node(std::make_shared<v8::Gather>(data, index, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
