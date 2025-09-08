// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_chunk_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto num_chunks = context.const_input<int>(1);
    auto dim = context.get_input(2);
    std::shared_ptr<ov::Node> chunk;

    auto shape = context.get_input(0).get_partial_shape();
    if (shape.rank().is_dynamic()) {
        size_t num_splits = context.get_decoder()->output_list_size();
        std::vector<int32_t> split_lengths_vec;
        for (size_t i = 0; i < num_splits - 1; i++) {
            split_lengths_vec.push_back(num_chunks);
        }
        split_lengths_vec.push_back(-1);
        auto split_lengths =
            context.mark_node(v0::Constant::create(element::i32, Shape{num_splits}, split_lengths_vec));
        auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(context.get_input(0), dim, split_lengths));
        return {context.mark_node(make_list_construct(split->outputs()))};
    }
    auto dim_val = context.const_input<int>(2);
    if (dim_val < 0) {
        dim_val = static_cast<int>(shape.rank().get_length()) + dim_val;
    }
    int num_splits = static_cast<int>(shape[dim_val].get_length()) / num_chunks;

    chunk = context.mark_node(std::make_shared<v1::Split>(context.get_input(0), dim, num_splits));

    return {context.mark_node(make_list_construct(chunk->outputs()))};
}

OutputVector translate_unbind_int_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    Output<Node> dim;
    int64_t dim_val = 0;
    if (context.input_is_none(1)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = context.get_input(1);
        dim_val = context.const_input<int>(1);
    }
    auto shape = input.get_shape();
    if (dim_val < 0) {
        dim_val = static_cast<int>(shape.size()) + dim_val;
    }

    auto num_splits = static_cast<int>(shape[dim_val]);
    auto chunk = context.mark_node(std::make_shared<v1::Split>(input, dim, num_splits));

    ov::OutputVector out_vec;
    for (auto& out : chunk->outputs())
        out_vec.push_back(std::make_shared<v0::Squeeze>(out, dim));

    return {context.mark_node(make_list_construct(out_vec))};
}

OutputVector translate_split_with_sizes(const NodeContext& context) {
    // aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
    num_inputs_check(context, 2, 3, true);
    auto data = context.get_input(0);
    auto split_lengths = get_input_concat_if_list(context, 1);
    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = context.get_input(2);
    }

    auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    if (is_complex) {
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, data, true));
        dim = normalize_axis(context, dim, rank);
        data = complex->get_input_source_output(0);
    }

    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(data, dim, split_lengths));

    auto res = split->outputs();
    if (is_complex) {
        for (auto& output : res) {
            output = context.mark_node(std::make_shared<ComplexTypeMark>(output));
        }
    }
    return {context.mark_node(make_list_construct(res))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
