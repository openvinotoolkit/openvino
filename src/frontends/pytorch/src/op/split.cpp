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

    // Determine the number of output tensors from the decoder.
    const size_t num_outputs = context.get_decoder()->output_list_size();
    PYTORCH_OP_CONVERSION_CHECK(num_outputs > 0,
                                "aten::unbind: cannot determine the number of outputs from the decoder.");

    auto split = context.mark_node(std::make_shared<v1::Split>(input, dim, num_outputs));

    ov::OutputVector outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(context.mark_node(std::make_shared<v0::Squeeze>(split->output(i), dim)));
    }
    return {context.mark_node(make_list_construct(wrap_complex(context, outputs, complex)))};
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

    // Complex inputs are split on their logical axes; unwrap here and re-wrap the results.
    auto complex = unwrap_complex_split(context, data, dim);
    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(data, dim, split_lengths));
    return {context.mark_node(make_list_construct(wrap_complex(context, split->outputs(), complex)))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
