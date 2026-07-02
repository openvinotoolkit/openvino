// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_chunk(const NodeContext& context) {
    // aten::chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
    num_inputs_check(context, 2, 3, true);
    auto input = context.get_input(0);
    auto chunks = get_input_as_i32(context, 1);

    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = get_input_as_i32(context, 2);
    }

    // Determine the number of output tensors from the decoder (static list size).
    const size_t num_outputs = context.get_decoder()->output_list_size();
    PYTORCH_OP_CONVERSION_CHECK(num_outputs > 0,
                                "aten::chunk: cannot determine the number of output chunks from the decoder.");

    if (num_outputs == 1) {
        return {context.mark_node(make_list_construct({input}))};
    }

    // Build a VariadicSplit: the first (num_outputs - 1) chunks have equal size
    // (= ceil(dim_size / num_chunks)) and the last chunk receives the remainder.
    auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto neg1_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto zero_sc = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    // Gather the size of the split dimension.
    auto dim_size = context.mark_node(std::make_shared<v8::Gather>(input_shape, dim, zero_sc));

    // chunk_size = ceil(dim_size / num_chunks)
    auto init_chunk_size = context.mark_node(std::make_shared<v1::Divide>(dim_size, chunks, true));
    auto last_chunk_mod = context.mark_node(std::make_shared<v1::Mod>(dim_size, chunks));
    auto is_nonzero = context.mark_node(std::make_shared<v1::Greater>(last_chunk_mod, zero_1d));
    auto is_nonzero_int = context.mark_node(std::make_shared<v0::Convert>(is_nonzero, element::i32));
    auto chunk_size = context.mark_node(std::make_shared<v1::Add>(init_chunk_size, is_nonzero_int));

    auto split_even_count = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {num_outputs - 1}));
    auto split_lengths_even = context.mark_node(std::make_shared<v3::Broadcast>(chunk_size, split_even_count));
    auto split_lengths = context.mark_node(std::make_shared<v0::Concat>(OutputVector{split_lengths_even, neg1_1d}, 0));

    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(input, dim, split_lengths));
    return {context.mark_node(make_list_construct(split->outputs()))};
}

OutputVector translate_unbind(const NodeContext& context) {
    // aten::unbind.int(Tensor self, int dim=0) -> Tensor[]
    num_inputs_check(context, 1, 2, true);
    auto input = context.get_input(0);

    Output<Node> dim;
    if (context.input_is_none(1)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = get_input_as_i32(context, 1);
    }

    // Normalise negative dim values.
    auto rank = std::get<1>(get_shape_rank(context, input, true));
    dim = normalize_axis(context, dim, rank);

    // Determine the number of output tensors from the decoder.
    const size_t num_outputs = context.get_decoder()->output_list_size();
    PYTORCH_OP_CONVERSION_CHECK(num_outputs > 0,
                                "aten::unbind: cannot determine the number of outputs from the decoder.");

    auto split = context.mark_node(std::make_shared<v1::Split>(input, dim, num_outputs));

    ov::OutputVector outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(context.mark_node(std::make_shared<v0::Squeeze>(split->output(i), dim)));
    }
    return {context.mark_node(make_list_construct(outputs))};
}

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
