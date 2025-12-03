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

OutputVector translate_index_select(const NodeContext& context) {
    // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
    // aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4, true);  // allow_complex = true
    auto x = context.get_input(0);
    auto dim = context.get_input(1);
    auto indices = context.get_input(2);

    auto complex = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex) {
        x = complex->get_input_source_output(0);

        // Normalize axis for complex tensor
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, context.get_input(0), true));
        dim = normalize_axis(context, dim, rank);
    }

    auto gather = context.mark_node(std::make_shared<v8::Gather>(x, indices, dim));
    if (!context.input_is_none(3)) {
        context.mutate_input(3, gather);
    }

    if (complex) {
        return {context.mark_node(std::make_shared<ComplexTypeMark>(gather, complex->get_complex_part_type()))};
    }
    return {gather};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
