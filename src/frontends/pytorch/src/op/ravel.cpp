// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_ravel(const NodeContext& context) {
    // torch.ravel(Tensor input) -> Tensor
    // The ravel operation is equivalent to flattening the tensor.
    num_inputs_check(context, 1, 1, true);
    auto tensor = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(tensor.get_node_shared_ptr());
    std::shared_ptr<Node> new_shape;
    if (complex_type_mark) {
        tensor = complex_type_mark->get_data();
        new_shape = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-1, 2}));
    } else {
        new_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    }

    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(tensor, new_shape, false));

    if (complex_type_mark) {
        const auto& complex_dtype = complex_type_mark->get_complex_part_type();
        return {context.mark_node(std::make_shared<ComplexTypeMark>(reshape, complex_dtype))};
    }
    return {reshape};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov