// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_reshape(const NodeContext& context) {
    // Translation is used by both aten::view and aten::reshape.
    // Schema: aten::view(Tensor input, int[] shape) -> Tensor
    // Schema: aten::reshape(Tensor input, int[] shape) -> Tensor
    // For shape parameter, int[] is converted into single dimensional Tensor.
    num_inputs_check(context, 2, 2, true);
    auto tensor = context.get_input(0);
    auto shape = get_input_concat_if_list(context, 1);

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
