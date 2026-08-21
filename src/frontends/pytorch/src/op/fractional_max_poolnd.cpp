// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
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

Output<Node> get_output_size_node(const NodeContext& context, int idx) {
    Output<Node> given_shape;
    auto shape_type = context.get_input_type(idx);
    if (shape_type.is<type::List>()) {
        const auto list_elems = get_list_as_outputs(context.get_input(idx));
        OutputVector to_concat;
        auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            for (auto elem : list_elems) {
                if (elem.get_element_type() != element::i32) {
                    elem = context.mark_node(std::make_shared<v0::Convert>(elem, element::i32));
                }
            to_concat.push_back(context.mark_node(std::make_shared<v0::Unsqueeze>(elem, zero)));
        }
        given_shape = context.mark_node(std::make_shared<v0::Concat>(to_concat, 0));
    } else {
        given_shape = get_input_as_i32(context, idx);
    }
    return given_shape;
}

Output<Node> generate_intervals(const NodeContext& context, Output<Node> input_size, Output<Node> output_size, Output<Node> pool_size, Output<Node> sample) {
    auto f32_type = element::f32;
    auto i32_type = element::i32;

    auto const_0 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {1}));

    auto out_minus_1 = context.mark_node(std::make_shared<v1::Subtract>(output_size, const_1)); // [1]
    auto in_minus_pool = context.mark_node(std::make_shared<v1::Subtract>(input_size, pool_size)); // [1]
    
    // alpha = (in - pool) / (out - 1)
    auto out_minus_1_f32 = context.mark_node(std::make_shared<v0::Convert>(out_minus_1, f32_type));
    auto in_minus_pool_f32 = context.mark_node(std::make_shared<v0::Convert>(in_minus_pool, f32_type));
    auto alpha = context.mark_node(std::make_shared<v1::Divide>(in_minus_pool_f32, out_minus_1_f32)); // [1]
    
    // i = arange(output_size - 1)
    auto i = context.mark_node(std::make_shared<v4::Range>(const_0, out_minus_1, const_1, f32_type)); // [o - 1]
    
    // sample * alpha -> [N, C, 1]
    auto sample_f32 = context.mark_node(std::make_shared<v0::Convert>(sample, f32_type));
    auto sample_alpha = context.mark_node(std::make_shared<v1::Multiply>(sample_f32, alpha));
    auto floor_sample_alpha = context.mark_node(std::make_shared<v0::Floor>(sample_alpha)); // [N, C, 1]
    
    // i + sample -> [N, C, o-1]
    auto i_plus_sample = context.mark_node(std::make_shared<v1::Add>(i, sample_f32));
    auto i_plus_sample_alpha = context.mark_node(std::make_shared<v1::Multiply>(i_plus_sample, alpha));
    auto floor_i_plus_sample_alpha = context.mark_node(std::make_shared<v0::Floor>(i_plus_sample_alpha)); // [N, C, o-1]
    
    auto sequence_body = context.mark_node(std::make_shared<v1::Subtract>(floor_i_plus_sample_alpha, floor_sample_alpha));
    sequence_body = context.mark_node(std::make_shared<v0::Convert>(sequence_body, i32_type)); // [N, C, o-1]
    
    auto sample_shape = context.mark_node(std::make_shared<v3::ShapeOf>(sample, i32_type));
    auto seq_last = context.mark_node(std::make_shared<v3::Broadcast>(in_minus_pool, sample_shape)); // [N, C, 1]
    
    auto sequence = context.mark_node(std::make_shared<v0::Concat>(OutputVector{sequence_body, seq_last}, 2)); // [N, C, o]
    
    // If output_size == 1, PyTorch avoids division by zero by just using input_size - pool_size
    auto is_one = context.mark_node(std::make_shared<v1::Equal>(output_size, const_1));
    return context.mark_node(std::make_shared<v1::Select>(is_one, seq_last, sequence));
}

}  // namespace

OutputVector translate_fractional_max_pool2d(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    
    auto input = context.get_input(0); // [N, C, H, W] or [C, H, W]
    auto random_samples = context.get_input(3); // [N, C, 2] or [C, 2]
    
    auto i32_type = element::i32;
    auto i64_type = element::i64;
    auto const_0 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {2}));
    auto const_3 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {3}));
    auto const_4 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {4}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {-1}));
    
    FRONT_END_OP_CONVERSION_CHECK(input.get_partial_shape().rank().is_static(),
                                  "fractional_max_pool2d requires static rank for input tensor.");
    auto rank = input.get_partial_shape().rank().get_length();
    FRONT_END_OP_CONVERSION_CHECK(rank == 3 || rank == 4,
                                  "fractional_max_pool2d expects input of rank 3 or 4.");
    
    bool is_3d = rank == 3;
    if (is_3d) {
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        random_samples = context.mark_node(std::make_shared<v0::Unsqueeze>(random_samples, const_0));
    }
    
    auto input_shape_4d = context.mark_node(std::make_shared<v3::ShapeOf>(input, i32_type));
    auto inputH = context.mark_node(std::make_shared<v8::Gather>(input_shape_4d, const_2, const_0)); // [1]
    auto inputW = context.mark_node(std::make_shared<v8::Gather>(input_shape_4d, const_3, const_0)); // [1]
    
    auto pool_size_node = get_output_size_node(context, 1);
    auto output_size_node = get_output_size_node(context, 2);
    
    auto kH = context.mark_node(std::make_shared<v8::Gather>(pool_size_node, const_0, const_0)); // [1]
    auto kW = context.mark_node(std::make_shared<v8::Gather>(pool_size_node, const_1, const_0)); // [1]
    
    auto oH = context.mark_node(std::make_shared<v8::Gather>(output_size_node, const_0, const_0)); // [1]
    auto oW = context.mark_node(std::make_shared<v8::Gather>(output_size_node, const_1, const_0)); // [1]
    
    // In PyTorch, random_samples index 0 is W and index 1 is H
    auto sampleW = context.mark_node(std::make_shared<v8::Slice>(random_samples, const_0, const_1, const_1, const_2)); // [N, C, 1]
    auto sampleH = context.mark_node(std::make_shared<v8::Slice>(random_samples, const_1, const_2, const_1, const_2)); // [N, C, 1]
    
    auto seqH = generate_intervals(context, inputH, oH, kH, sampleH); // [N, C, oH]
    auto seqW = generate_intervals(context, inputW, oW, kW, sampleW); // [N, C, oW]
    
    auto seqH_4d = context.mark_node(std::make_shared<v0::Unsqueeze>(seqH, const_3)); // [N, C, oH, 1]
    auto seqW_4d = context.mark_node(std::make_shared<v0::Unsqueeze>(seqW, const_2)); // [N, C, 1, oW]
    
    auto seqH_times_inW = context.mark_node(std::make_shared<v1::Multiply>(seqH_4d, inputW)); // [N, C, oH, 1]
    auto start_idx = context.mark_node(std::make_shared<v1::Add>(seqH_times_inW, seqW_4d)); // [N, C, oH, oW]
    
    auto offset_h = context.mark_node(std::make_shared<v4::Range>(const_0, kH, const_1, i32_type));
    auto offset_w = context.mark_node(std::make_shared<v4::Range>(const_0, kW, const_1, i32_type));
    
    auto offset_h_2d = context.mark_node(std::make_shared<v0::Unsqueeze>(offset_h, const_1)); // [kH, 1]
    auto offset_w_2d = context.mark_node(std::make_shared<v0::Unsqueeze>(offset_w, const_0)); // [1, kW]
    
    auto offset_h_times_inW = context.mark_node(std::make_shared<v1::Multiply>(offset_h_2d, inputW));
    auto window_offsets_2d = context.mark_node(std::make_shared<v1::Add>(offset_h_times_inW, offset_w_2d)); // [kH, kW]
    
    auto window_offsets = context.mark_node(std::make_shared<v1::Reshape>(window_offsets_2d, const_neg_1, false)); // [kH * kW]
    
    auto start_idx_5d = context.mark_node(std::make_shared<v0::Unsqueeze>(start_idx, const_4)); // [N, C, oH, oW, 1]
    auto gather_indices = context.mark_node(std::make_shared<v1::Add>(start_idx_5d, window_offsets)); // [N, C, oH, oW, kH*kW]
    
    auto N_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape_4d, const_0, const_0));
    auto C_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape_4d, const_1, const_0));
    auto input_flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{N_dim, C_dim, const_neg_1}, 0));
    auto input_flat = context.mark_node(std::make_shared<v1::Reshape>(input, input_flat_shape, false)); // [N, C, inH*inW]
    
    // gather patches: data [N, C, L], indices [N, C, oH, oW, K], axis 2, batch_dims 2
    auto gathered = context.mark_node(std::make_shared<v8::Gather>(input_flat, gather_indices, const_2, 2)); // [N, C, oH, oW, kH*kW]
    
    // Use SORT_INDICES to ensure stable tie-breaking that matches PyTorch's forward scan (first index wins on ties)
    auto topk = context.mark_node(std::make_shared<v11::TopK>(gathered, const_1, const_4, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_INDICES, i32_type, true));
    auto max_values = topk->output(0); // [N, C, oH, oW, 1]
    auto max_indices_within_window = topk->output(1); // [N, C, oH, oW, 1]
    
    auto local_offset = context.mark_node(std::make_shared<v8::Gather>(window_offsets, max_indices_within_window, const_0)); // [N, C, oH, oW, 1]
    auto original_indices = context.mark_node(std::make_shared<v1::Add>(start_idx_5d, local_offset)); // [N, C, oH, oW, 1]
    
    auto pooled_tensor = context.mark_node(std::make_shared<v0::Squeeze>(max_values, const_4)); // [N, C, oH, oW]
    auto pooled_indices = context.mark_node(std::make_shared<v0::Squeeze>(original_indices, const_4)); // [N, C, oH, oW]
    
    pooled_indices = context.mark_node(std::make_shared<v0::Convert>(pooled_indices, i64_type));
    
    if (is_3d) {
        pooled_tensor = context.mark_node(std::make_shared<v0::Squeeze>(pooled_tensor, const_0)); // [C, oH, oW]
        pooled_indices = context.mark_node(std::make_shared<v0::Squeeze>(pooled_indices, const_0));
    }
    
    return {std::move(pooled_tensor), std::move(pooled_indices)};
};

OutputVector translate_fractional_max_pool3d(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    
    auto input = context.get_input(0); // [N, C, D, H, W] or [C, D, H, W]
    auto random_samples = context.get_input(3); // [N, C, 3] or [C, 3]
    
    auto i32_type = element::i32;
    auto i64_type = element::i64;
    auto const_0 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {2}));
    auto const_3 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {3}));
    auto const_4 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {4}));
    auto const_5 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {5}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(i32_type, Shape{1}, {-1}));
    
    FRONT_END_OP_CONVERSION_CHECK(input.get_partial_shape().rank().is_static(),
                                  "fractional_max_pool3d requires static rank for input tensor.");
    auto rank = input.get_partial_shape().rank().get_length();
    FRONT_END_OP_CONVERSION_CHECK(rank == 4 || rank == 5,
                                  "fractional_max_pool3d expects input of rank 4 or 5.");
    
    bool is_4d = rank == 4;
    if (is_4d) {
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        random_samples = context.mark_node(std::make_shared<v0::Unsqueeze>(random_samples, const_0));
    }
    
    auto input_shape_5d = context.mark_node(std::make_shared<v3::ShapeOf>(input, i32_type));
    auto inputD = context.mark_node(std::make_shared<v8::Gather>(input_shape_5d, const_2, const_0)); // [1]
    auto inputH = context.mark_node(std::make_shared<v8::Gather>(input_shape_5d, const_3, const_0)); // [1]
    auto inputW = context.mark_node(std::make_shared<v8::Gather>(input_shape_5d, const_4, const_0)); // [1]
    
    auto pool_size_node = get_output_size_node(context, 1);
    auto output_size_node = get_output_size_node(context, 2);
    
    auto kD = context.mark_node(std::make_shared<v8::Gather>(pool_size_node, const_0, const_0));
    auto kH = context.mark_node(std::make_shared<v8::Gather>(pool_size_node, const_1, const_0));
    auto kW = context.mark_node(std::make_shared<v8::Gather>(pool_size_node, const_2, const_0));
    
    auto oD = context.mark_node(std::make_shared<v8::Gather>(output_size_node, const_0, const_0));
    auto oH = context.mark_node(std::make_shared<v8::Gather>(output_size_node, const_1, const_0));
    auto oW = context.mark_node(std::make_shared<v8::Gather>(output_size_node, const_2, const_0));
    
    // In PyTorch 3D: index 0 is T(D), index 1 is H, index 2 is W
    auto sampleD = context.mark_node(std::make_shared<v8::Slice>(random_samples, const_0, const_1, const_1, const_2));
    auto sampleH = context.mark_node(std::make_shared<v8::Slice>(random_samples, const_1, const_2, const_1, const_2));
    auto sampleW = context.mark_node(std::make_shared<v8::Slice>(random_samples, const_2, const_3, const_1, const_2));
    
    auto seqD = generate_intervals(context, inputD, oD, kD, sampleD); // [N, C, oD]
    auto seqH = generate_intervals(context, inputH, oH, kH, sampleH); // [N, C, oH]
    auto seqW = generate_intervals(context, inputW, oW, kW, sampleW); // [N, C, oW]
    
    auto seqD_5d = context.mark_node(std::make_shared<v0::Unsqueeze>(seqD, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {3, 4})))); // [N, C, oD, 1, 1]
    auto seqH_5d = context.mark_node(std::make_shared<v0::Unsqueeze>(seqH, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {2, 4})))); // [N, C, 1, oH, 1]
    auto seqW_5d = context.mark_node(std::make_shared<v0::Unsqueeze>(seqW, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {2, 3})))); // [N, C, 1, 1, oW]
    
    auto inH_times_inW = context.mark_node(std::make_shared<v1::Multiply>(inputH, inputW));
    auto seqD_times_hw = context.mark_node(std::make_shared<v1::Multiply>(seqD_5d, inH_times_inW));
    auto seqH_times_w = context.mark_node(std::make_shared<v1::Multiply>(seqH_5d, inputW));
    
    auto start_idx_dh = context.mark_node(std::make_shared<v1::Add>(seqD_times_hw, seqH_times_w));
    auto start_idx = context.mark_node(std::make_shared<v1::Add>(start_idx_dh, seqW_5d)); // [N, C, oD, oH, oW]
    
    auto offset_d = context.mark_node(std::make_shared<v4::Range>(const_0, kD, const_1, i32_type));
    auto offset_h = context.mark_node(std::make_shared<v4::Range>(const_0, kH, const_1, i32_type));
    auto offset_w = context.mark_node(std::make_shared<v4::Range>(const_0, kW, const_1, i32_type));
    
    auto offset_d_3d = context.mark_node(std::make_shared<v0::Unsqueeze>(offset_d, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {1, 2})))); // [kD, 1, 1]
    auto offset_h_3d = context.mark_node(std::make_shared<v0::Unsqueeze>(offset_h, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {0, 2})))); // [1, kH, 1]
    auto offset_w_3d = context.mark_node(std::make_shared<v0::Unsqueeze>(offset_w, context.mark_node(v0::Constant::create(i32_type, Shape{2}, {0, 1})))); // [1, 1, kW]
    
    auto offset_d_times_hw = context.mark_node(std::make_shared<v1::Multiply>(offset_d_3d, inH_times_inW));
    auto offset_h_times_w = context.mark_node(std::make_shared<v1::Multiply>(offset_h_3d, inputW));
    
    auto window_offsets_dh = context.mark_node(std::make_shared<v1::Add>(offset_d_times_hw, offset_h_times_w));
    auto window_offsets_3d = context.mark_node(std::make_shared<v1::Add>(window_offsets_dh, offset_w_3d)); // [kD, kH, kW]
    
    auto window_offsets = context.mark_node(std::make_shared<v1::Reshape>(window_offsets_3d, const_neg_1, false)); // [kD * kH * kW]
    
    auto start_idx_6d = context.mark_node(std::make_shared<v0::Unsqueeze>(start_idx, const_5)); // [N, C, oD, oH, oW, 1]
    auto gather_indices = context.mark_node(std::make_shared<v1::Add>(start_idx_6d, window_offsets)); // [N, C, oD, oH, oW, kD*kH*kW]
    
    auto N_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape_5d, const_0, const_0));
    auto C_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape_5d, const_1, const_0));
    auto input_flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{N_dim, C_dim, const_neg_1}, 0));
    auto input_flat = context.mark_node(std::make_shared<v1::Reshape>(input, input_flat_shape, false)); // [N, C, inD*inH*inW]
    
    auto gathered = context.mark_node(std::make_shared<v8::Gather>(input_flat, gather_indices, const_2, 2)); // [N, C, oD, oH, oW, kD*kH*kW]
    
    // Use SORT_INDICES to ensure stable tie-breaking that matches PyTorch's forward scan (first index wins on ties)
    auto topk = context.mark_node(std::make_shared<v11::TopK>(gathered, const_1, const_5, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_INDICES, i32_type, true));
    auto max_values = topk->output(0); // [N, C, oD, oH, oW, 1]
    auto max_indices_within_window = topk->output(1); // [N, C, oD, oH, oW, 1]
    
    auto local_offset = context.mark_node(std::make_shared<v8::Gather>(window_offsets, max_indices_within_window, const_0));
    auto original_indices = context.mark_node(std::make_shared<v1::Add>(start_idx_6d, local_offset));
    
    auto pooled_tensor = context.mark_node(std::make_shared<v0::Squeeze>(max_values, const_5)); // [N, C, oD, oH, oW]
    auto pooled_indices = context.mark_node(std::make_shared<v0::Squeeze>(original_indices, const_5)); // [N, C, oD, oH, oW]
    
    pooled_indices = context.mark_node(std::make_shared<v0::Convert>(pooled_indices, i64_type));
    
    if (is_4d) {
        pooled_tensor = context.mark_node(std::make_shared<v0::Squeeze>(pooled_tensor, const_0)); // [C, oD, oH, oW]
        pooled_indices = context.mark_node(std::make_shared<v0::Squeeze>(pooled_indices, const_0));
    }
    
    return {std::move(pooled_tensor), std::move(pooled_indices)};
};

OutputVector translate_fractional_max_pool3d_fx(const NodeContext& context) {
    auto outs = translate_fractional_max_pool3d(context);
    return {context.mark_node(make_list_construct(outs))};
};

OutputVector translate_fractional_max_pool2d_fx(const NodeContext& context) {
    auto outs = translate_fractional_max_pool2d(context);
    return {context.mark_node(make_list_construct(outs))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
