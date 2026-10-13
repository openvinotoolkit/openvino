// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/greater_equal.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

Output<Node> get_int_list_input(const NodeContext& context, size_t idx) {
    if (context.get_input_type(idx).is<type::List>()) {
        const auto list_elems = get_list_as_outputs(context.get_input(static_cast<int>(idx)));
        OutputVector to_concat;
        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
        for (auto elem : list_elems) {
            if (elem.get_element_type() != element::i32) {
                elem = context.mark_node(std::make_shared<v0::Convert>(elem, element::i32));
            }
            to_concat.push_back(context.mark_node(std::make_shared<v0::Unsqueeze>(elem, zero)));
        }
        return context.mark_node(std::make_shared<v0::Concat>(to_concat, 0));
    }
    return get_input_as_i32(context, idx);
}

Output<Node> generate_intervals(const NodeContext& context,
                                const Output<Node>& input_size,
                                const Output<Node>& output_size,
                                const Output<Node>& pool_size,
                                const Output<Node>& sample) {
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));

    auto out_minus_one = context.mark_node(std::make_shared<v1::Subtract>(output_size, one));
    auto in_minus_pool = context.mark_node(std::make_shared<v1::Subtract>(input_size, pool_size));
    auto safe_denominator = context.mark_node(std::make_shared<v1::Maximum>(out_minus_one, one));

    auto numerator = context.mark_node(std::make_shared<v1::ConvertLike>(in_minus_pool, sample));
    auto denominator = context.mark_node(std::make_shared<v1::ConvertLike>(safe_denominator, sample));
    auto alpha = context.mark_node(std::make_shared<v1::Divide>(numerator, denominator));

    auto steps = context.mark_node(std::make_shared<v4::Range>(zero, out_minus_one, one, sample.get_element_type()));
    auto steps_plus_sample = context.mark_node(std::make_shared<v1::Add>(steps, sample));
    auto scaled_steps = context.mark_node(std::make_shared<v1::Multiply>(steps_plus_sample, alpha));
    auto floor_scaled_steps = context.mark_node(std::make_shared<v0::Floor>(scaled_steps));
    auto scaled_sample = context.mark_node(std::make_shared<v1::Multiply>(sample, alpha));
    auto floor_scaled_sample = context.mark_node(std::make_shared<v0::Floor>(scaled_sample));
    Output<Node> body = context.mark_node(std::make_shared<v1::Subtract>(floor_scaled_steps, floor_scaled_sample));
    body = context.mark_node(std::make_shared<v0::Convert>(body, element::i32));

    auto sample_shape = context.mark_node(std::make_shared<v3::ShapeOf>(sample, element::i32));
    auto last = context.mark_node(std::make_shared<v3::Broadcast>(in_minus_pool, sample_shape));

    return context.mark_node(std::make_shared<v0::Concat>(OutputVector{body, last}, -1));
}

std::vector<int64_t> all_axes_except(int64_t count, int64_t offset, int64_t keep) {
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < count; ++i) {
        if (i != keep) {
            axes.push_back(offset + i);
        }
    }
    return axes;
}

OutputVector translate_fractional_max_pool_nd(const NodeContext& context,
                                              int64_t spatial_rank,
                                              const std::vector<int64_t>& sample_index_per_dim) {
    num_inputs_check(context, 4, 4);
    auto input = context.get_input(0);
    auto random_samples = context.get_input(3);

    const auto input_rank = input.get_partial_shape().rank();
    PYTORCH_OP_CONVERSION_CHECK(input_rank.is_static(),
                                "fractional_max_pool",
                                spatial_rank,
                                "d requires input with static rank.");
    const auto rank = input_rank.get_length();
    PYTORCH_OP_CONVERSION_CHECK(rank == spatial_rank + 1 || rank == spatial_rank + 2,
                                "fractional_max_pool",
                                spatial_rank,
                                "d expects input of rank ",
                                spatial_rank + 1,
                                " or ",
                                spatial_rank + 2,
                                ".");

    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto samples_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
    auto slice_step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    const bool unbatched = rank == spatial_rank + 1;
    if (unbatched) {
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, zero));
    }

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto pool_size = get_int_list_input(context, 1);
    auto output_size = get_int_list_input(context, 2);

    OutputVector input_dims;
    OutputVector pool_dims;
    OutputVector output_dims;
    for (int64_t d = 0; d < spatial_rank; ++d) {
        auto dim_idx = context.mark_node(v0::Constant::create(element::i32, Shape{}, {d}));
        auto input_dim_idx = context.mark_node(v0::Constant::create(element::i32, Shape{}, {d + 2}));
        input_dims.push_back(context.mark_node(std::make_shared<v8::Gather>(input_shape, input_dim_idx, zero)));
        pool_dims.push_back(context.mark_node(std::make_shared<v8::Gather>(pool_size, dim_idx, zero)));
        output_dims.push_back(context.mark_node(std::make_shared<v8::Gather>(output_size, dim_idx, zero)));
    }

    OutputVector strides(spatial_rank);
    strides[spatial_rank - 1] = one;
    for (int64_t d = spatial_rank - 2; d >= 0; --d) {
        strides[d] = context.mark_node(std::make_shared<v1::Multiply>(strides[d + 1], input_dims[d + 1]));
    }

    const auto expand_axes_shape = Shape{static_cast<size_t>(spatial_rank - 1)};
    Output<Node> start_idx;
    Output<Node> window_offsets;
    Output<Node> window_size = one;
    for (int64_t d = 0; d < spatial_rank; ++d) {
        const auto sample_idx = sample_index_per_dim[d];
        auto sample_begin = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {sample_idx}));
        auto sample_end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {sample_idx + 1}));
        auto sample = context.mark_node(
            std::make_shared<v8::Slice>(random_samples, sample_begin, sample_end, slice_step, samples_axis));

        auto seq = generate_intervals(context, input_dims[d], output_dims[d], pool_dims[d], sample);
        auto seq_scaled = context.mark_node(std::make_shared<v1::Multiply>(seq, strides[d]));
        auto seq_axes = context.mark_node(
            v0::Constant::create(element::i64, expand_axes_shape, all_axes_except(spatial_rank, 2, d)));
        Output<Node> seq_expanded = context.mark_node(std::make_shared<v0::Unsqueeze>(seq_scaled, seq_axes));
        if (d == 0) {
            start_idx = seq_expanded;
        } else {
            start_idx = context.mark_node(std::make_shared<v1::Add>(start_idx, seq_expanded));
        }

        auto offsets = context.mark_node(std::make_shared<v4::Range>(zero, pool_dims[d], one, element::i32));
        auto offsets_scaled = context.mark_node(std::make_shared<v1::Multiply>(offsets, strides[d]));
        auto offset_axes = context.mark_node(
            v0::Constant::create(element::i64, expand_axes_shape, all_axes_except(spatial_rank, 0, d)));
        Output<Node> offsets_expanded = context.mark_node(std::make_shared<v0::Unsqueeze>(offsets_scaled, offset_axes));
        if (d == 0) {
            window_offsets = offsets_expanded;
        } else {
            window_offsets = context.mark_node(std::make_shared<v1::Add>(window_offsets, offsets_expanded));
        }
        window_size = context.mark_node(std::make_shared<v1::Multiply>(window_size, pool_dims[d]));
    }

    auto flat_window_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    window_offsets = context.mark_node(std::make_shared<v1::Reshape>(window_offsets, flat_window_shape, false));

    auto window_axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {spatial_rank + 2}));
    auto start_idx_expanded = context.mark_node(std::make_shared<v0::Unsqueeze>(start_idx, window_axis));
    auto gather_indices = context.mark_node(std::make_shared<v1::Add>(start_idx_expanded, window_offsets));

    auto flat_input_shape = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {0, 0, -1}));
    auto input_flat = context.mark_node(std::make_shared<v1::Reshape>(input, flat_input_shape, true));
    auto flat_axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    auto windows = context.mark_node(std::make_shared<v8::Gather>(input_flat, gather_indices, flat_axis, 2));

    auto is_nan = context.mark_node(std::make_shared<v1::NotEqual>(windows, windows));
    auto neg_inf =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
    auto neg_inf_like = context.mark_node(std::make_shared<v1::ConvertLike>(neg_inf, windows));
    auto windows_no_nan = context.mark_node(std::make_shared<v1::Select>(is_nan, neg_inf_like, windows));

    auto k = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto topk = context.mark_node(std::make_shared<v11::TopK>(windows_no_nan,
                                                              k,
                                                              -1,
                                                              v11::TopK::Mode::MAX,
                                                              v11::TopK::SortType::SORT_VALUES,
                                                              element::i32,
                                                              true));

    auto positions = context.mark_node(std::make_shared<v4::Range>(zero, window_size, one, element::i32));
    auto no_position = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto nan_positions = context.mark_node(std::make_shared<v1::Select>(is_nan, positions, no_position));
    auto reduce_axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto last_nan = context.mark_node(std::make_shared<v1::ReduceMax>(nan_positions, reduce_axis, true));
    auto has_nan = context.mark_node(std::make_shared<v1::GreaterEqual>(last_nan, zero));
    auto local_idx = context.mark_node(std::make_shared<v1::Select>(has_nan, last_nan, topk->output(1)));

    auto pooled = context.mark_node(std::make_shared<v6::GatherElements>(windows, local_idx, -1));
    auto local_offset = context.mark_node(std::make_shared<v8::Gather>(window_offsets, local_idx, zero));
    auto indices = context.mark_node(std::make_shared<v1::Add>(start_idx_expanded, local_offset));

    Output<Node> pooled_out = context.mark_node(std::make_shared<v0::Squeeze>(pooled, window_axis));
    Output<Node> indices_out = context.mark_node(std::make_shared<v0::Squeeze>(indices, window_axis));
    indices_out = context.mark_node(std::make_shared<v0::Convert>(indices_out, element::i64));

    if (unbatched) {
        pooled_out = context.mark_node(std::make_shared<v0::Squeeze>(pooled_out, zero));
        indices_out = context.mark_node(std::make_shared<v0::Squeeze>(indices_out, zero));
    }
    return {pooled_out, indices_out};
}

}  // namespace

OutputVector translate_fractional_max_pool2d(const NodeContext& context) {
    return translate_fractional_max_pool_nd(context, 2, {1, 0});
}

OutputVector translate_fractional_max_pool3d(const NodeContext& context) {
    return translate_fractional_max_pool_nd(context, 3, {0, 1, 2});
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
