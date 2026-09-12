// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_ravel(const NodeContext& context) {
    // Schema: aten::ravel(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    auto tensor = context.get_input(0);
    Output<Node> shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(tensor.get_node_shared_ptr());
    if (complex_type_mark) {
        tensor = complex_type_mark->get_data();
        auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
        const_2 = context.mark_node(std::make_shared<v1::ConvertLike>(const_2, shape));
        shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{shape, const_2}, 0));
    }

    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(tensor, shape, false));

    if (complex_type_mark) {
        const auto& complex_dtype = complex_type_mark->get_complex_part_type();
        return {context.mark_node(std::make_shared<ComplexTypeMark>(reshape, complex_dtype))};
    } else {
        return {reshape};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
