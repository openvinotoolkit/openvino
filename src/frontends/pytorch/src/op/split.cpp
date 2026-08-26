// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
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

namespace {
/// \brief Prepares a possibly-complex tensor for a split-like op.
///
/// When \p data is a ComplexTypeMark, \p dim is converted to i32 and normalized against the logical
/// rank so negative axes never target the encoded real/imag dimension, and \p data is replaced by
/// the underlying real tensor. Both \p data and \p dim are modified in place. When \p data is not
/// complex, \p data and \p dim are left unchanged.
///
/// \param context Node context used for marking nodes.
/// \param data Input tensor, unwrapped in place when complex.
/// \param dim Split axis, converted to i32 and normalized in place when the input is complex.
/// \return The ComplexTypeMark node when \p data was complex, otherwise nullptr; pass it to
///         wrap_complex to re-wrap the split results.
std::shared_ptr<ComplexTypeMark> unwrap_complex_split(const NodeContext& context,
                                                      Output<Node>& data,
                                                      Output<Node>& dim) {
    auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    if (complex) {
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, data, true));
        dim = normalize_axis(context, dim, rank);
        data = complex->get_input_source_output(0);
    }
    return complex;
}
}  // namespace

namespace {
size_t get_effective_num_outputs(const NodeContext& context,
                                const Output<Node>& input,
                                const Output<Node>& dim,
                                bool is_chunk,
                                bool is_split_by_size = false) {
    const size_t decoder_count = context.get_decoder()->output_list_size();
    if (decoder_count > 0) {
        return decoder_count;
    }

    auto shape = input.get_partial_shape();
    if (shape.rank().is_dynamic()) {
        shape = context.get_decoder()->get_input_shape(0);
    }
    if (!shape.rank().is_static()) {
        return 0;
    }

    int64_t dim_val = 0;
    if (is_chunk || is_split_by_size) {
        dim_val = context.input_is_none(2) ? 0 : context.const_input<int>(2);
    } else {
        dim_val = context.input_is_none(1) ? 0 : context.const_input<int>(1);
    }

    const auto rank = shape.rank().get_length();
    if (dim_val < 0) {
        dim_val += static_cast<int64_t>(rank);
    }
    if (dim_val < 0 || dim_val >= static_cast<int64_t>(rank) || !shape[dim_val].is_static()) {
        return 0;
    }

    if (is_chunk) {
        const auto dim_size = static_cast<size_t>(shape[dim_val].get_length());
        const auto requested_chunks = context.const_input<int64_t>(1);
        if (requested_chunks <= 0) {
            return 0;
        }
        if (dim_size == 0) {
            return static_cast<size_t>(requested_chunks);
        }
        const auto chunk_size = 1 + (dim_size - 1) / static_cast<size_t>(requested_chunks);
        return 1 + (dim_size - 1) / chunk_size;
    }
    if (is_split_by_size) {
        const auto dim_size = static_cast<size_t>(shape[dim_val].get_length());
        const auto split_size = context.const_input<int64_t>(1);
        if (dim_size == 0) {
            return split_size >= 0 ? 1 : 0;
        }
        if (split_size <= 0) {
            return 0;
        }
        return 1 + (dim_size - 1) / static_cast<size_t>(split_size);
    }
    return static_cast<size_t>(shape[dim_val].get_length());
}
}  // namespace

OutputVector translate_chunk(const NodeContext& context) {
    // aten::chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
    num_inputs_check(context, 2, 3, true);
    auto input = context.get_input(0);
    const auto num_chunks = context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(num_chunks > 0, "aten::chunk: chunks must be greater than zero.");

    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = get_input_as_i32(context, 2);
    }

    auto complex = unwrap_complex_split(context, input, dim);
    if (!complex) {
        auto rank = std::get<1>(get_shape_rank(context, input, true));
        dim = normalize_axis(context, dim, rank);
    }

    auto shape = input.get_partial_shape();
    if (shape.rank().is_dynamic()) {
        shape = context.get_decoder()->get_input_shape(0);
    }
    const size_t num_outputs = get_effective_num_outputs(context, input, dim, true);
    PYTORCH_OP_CONVERSION_CHECK(num_outputs > 0 && shape.rank().is_static(),
                                "aten::chunk: cannot determine the number of outputs from the input shape.");

    int64_t dim_val = context.input_is_none(2) ? 0 : context.const_input<int64_t>(2);
    if (dim_val < 0) {
        dim_val += shape.rank().get_length();
    }
    PYTORCH_OP_CONVERSION_CHECK(dim_val >= 0 && dim_val < shape.rank().get_length(),
                                "aten::chunk: dimension is out of range for the input rank.");
    PYTORCH_OP_CONVERSION_CHECK(shape[dim_val].is_static(),
                                "aten::chunk: the split dimension must be static.");
    const auto dim_size = static_cast<size_t>(shape[dim_val].get_length());
    const auto requested_chunks = static_cast<size_t>(num_chunks);
    const auto chunk_size = dim_size == 0 ? 0 : 1 + (dim_size - 1) / requested_chunks;

    std::vector<int64_t> split_lengths_vec(num_outputs - 1, static_cast<int64_t>(chunk_size));
    split_lengths_vec.push_back(-1);
    auto split_lengths = context.mark_node(
        v0::Constant::create(element::i64, Shape{num_outputs}, split_lengths_vec));
    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(input, dim, split_lengths));
    return {context.mark_node(make_list_construct(wrap_complex(context, split->outputs(), complex)))};
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

    // Complex inputs are split on their logical axes; unwrap here and re-wrap the results.
    auto complex = unwrap_complex_split(context, input, dim);
    if (!complex) {
        // Normalise negative dim values against the input rank.
        auto rank = std::get<1>(get_shape_rank(context, input, true));
        dim = normalize_axis(context, dim, rank);
    }

    // output_list_size() reflects requested getitem indices, not the real list arity.
    // Fallback to the shape-derived arity when the decoder cannot prove a true list unpack.
    const size_t num_outputs = get_effective_num_outputs(context, input, dim, false);
    PYTORCH_OP_CONVERSION_CHECK(num_outputs > 0,
                                "aten::unbind: cannot determine the number of outputs from the decoder or input shape.");

    auto split = context.mark_node(std::make_shared<v1::Split>(input, dim, num_outputs));

    ov::OutputVector outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(context.mark_node(std::make_shared<v0::Squeeze>(split->output(i), dim)));
    }
    return {context.mark_node(make_list_construct(wrap_complex(context, outputs, complex)))};
}

OutputVector translate_chunk_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    const auto split_size = context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(split_size >= 0, "aten::split.Tensor: split_size must be non-negative.");
    auto dim = context.get_input(2);

    const size_t num_splits = get_effective_num_outputs(context, context.get_input(0), dim, false, true);
    PYTORCH_OP_CONVERSION_CHECK(num_splits > 0,
                                "aten::split.Tensor: cannot determine the number of outputs from the input shape.");

    std::vector<int64_t> split_lengths_vec(num_splits - 1, split_size);
    split_lengths_vec.push_back(-1);
    auto split_lengths = context.mark_node(
        v0::Constant::create(element::i64, Shape{num_splits}, split_lengths_vec));
    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(context.get_input(0), dim, split_lengths));

    return {context.mark_node(make_list_construct(split->outputs()))};
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

    // Complex inputs are split on their logical axes; unwrap here and re-wrap the results.
    auto complex = unwrap_complex_split(context, data, dim);
    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(data, dim, split_lengths));
    return {context.mark_node(make_list_construct(wrap_complex(context, split->outputs(), complex)))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
