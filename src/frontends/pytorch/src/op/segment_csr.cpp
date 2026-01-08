// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

#include <limits>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_segment_mean_csr(const NodeContext& context) {
    // torch_scatter::segment_mean_csr(src, indptr, out=None)
    num_inputs_check(context, 2, 3);
    auto src = context.get_input(0);
    auto indptr = context.get_input(1);

    // Convert indptr to i32 to avoid type mismatch with shift (which is derived from shape, usually i32 or i64, but we force i32)
    // Actually, ShapeOf defaults to i64? No, get_shape_rank default uses i32 if specified, but let's check.
    // In our code: mark_node(std::make_shared<v3::ShapeOf>(src, element::i32));
    // So shapes are i32. indptr might be i64 (LongTensor).
    // Safe to convert to i32 if we assume indices fit.
    // If indptr is i64, we should convert to i32 for consistency with our shape computations.
    indptr = context.mark_node(std::make_shared<v0::Convert>(indptr, element::i32));

    // Get ranks and shapes
    auto src_shape = context.mark_node(std::make_shared<v3::ShapeOf>(src, element::i32));
    auto indptr_shape = context.mark_node(std::make_shared<v3::ShapeOf>(indptr, element::i32));

    // We need rank of indptr to determine reduction dim
    auto indptr_rank_node = std::get<1>(get_shape_rank(context, indptr));
    auto one_node = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto k_node = context.mark_node(std::make_shared<v1::Subtract>(indptr_rank_node, one_node));

    // Use Reshape to make it 1D for Slicing
    auto k_vec = context.mark_node(std::make_shared<v1::Reshape>(k_node, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));

    auto zero_node = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto one_node_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));

    auto src_prefix = context.mark_node(std::make_shared<v8::Slice>(src_shape, zero_node, k_vec, one_node_1d));

    auto k_plus_one = context.mark_node(std::make_shared<v1::Add>(k_node, one_node));
    auto k_plus_one_vec = context.mark_node(std::make_shared<v1::Reshape>(k_plus_one, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));

    auto max_int = std::numeric_limits<int32_t>::max();
    auto max_node = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {max_int}));

    auto src_suffix = context.mark_node(std::make_shared<v8::Slice>(src_shape, k_plus_one_vec, max_node, one_node_1d));

    // Broadcast indptr to match src_prefix + [Y]
    auto indptr_last_dim = context.mark_node(std::make_shared<v8::Slice>(indptr_shape, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1})), max_node, one_node_1d));

    auto target_indptr_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{src_prefix, indptr_last_dim}, 0));

    auto indptr_broadcasted = context.mark_node(std::make_shared<v3::Broadcast>(indptr, target_indptr_shape));

    // Flattening
    auto axes_node = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto batch_size = context.mark_node(std::make_shared<v1::ReduceProd>(src_prefix, axes_node, false));
    auto feature_size = context.mark_node(std::make_shared<v1::ReduceProd>(src_suffix, axes_node, false));

    // Get X (src dim K) and Y (indptr last dim)
    auto X = context.mark_node(std::make_shared<v8::Gather>(src_shape, k_vec, axes_node));
    auto Y = context.mark_node(std::make_shared<v8::Gather>(indptr_shape, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1})), axes_node));

    auto batch_size_1d = context.mark_node(std::make_shared<v1::Reshape>(batch_size, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));
    auto feature_size_1d = context.mark_node(std::make_shared<v1::Reshape>(feature_size, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));
    auto X_1d = context.mark_node(std::make_shared<v1::Reshape>(X, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));
    auto Y_1d = context.mark_node(std::make_shared<v1::Reshape>(Y, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));

    auto src_flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_size_1d, X_1d, feature_size_1d}, 0));
    auto src_flat = context.mark_node(std::make_shared<v1::Reshape>(src, src_flat_shape, false));

    // Reshape indptr to [Batch, Y]
    auto indptr_flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_size_1d, Y_1d}, 0));
    auto indptr_flat = context.mark_node(std::make_shared<v1::Reshape>(indptr_broadcasted, indptr_flat_shape, false));

    // Prepare EmbeddingBag inputs
    // Use FULL indptr as offsets. This creates bags:
    // [p0, p1), [p1, p2), ..., [p_{Y-1}, End)
    // We want the first Y-1 bags.
    // The last bag [p_{Y-1}, End) corresponds to "tail" elements.

    auto offsets_local = indptr_flat; // [Batch, Y]

    // Create shifts
    auto range_start = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto range_step = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto batch_idx = context.mark_node(std::make_shared<v4::Range>(range_start, batch_size, range_step, element::i32));

    auto batch_idx_reshaped = context.mark_node(std::make_shared<v1::Reshape>(batch_idx, context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-1, 1})), false));
    auto shift = context.mark_node(std::make_shared<v1::Multiply>(batch_idx_reshaped, X));

    // offsets_local is [B, Y]. shift is [B, 1].
    auto offsets_global_2d = context.mark_node(std::make_shared<v1::Add>(offsets_local, shift));

    auto offsets_global = context.mark_node(std::make_shared<v1::Reshape>(offsets_global_2d, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1})), false));

    auto total_elements = context.mark_node(std::make_shared<v1::Multiply>(batch_size, X));
    auto indices = context.mark_node(std::make_shared<v4::Range>(range_start, total_elements, range_step, element::i32));

    auto emb_table_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{
        context.mark_node(std::make_shared<v1::Reshape>(total_elements, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false)),
        feature_size_1d
    }, 0));
    auto emb_table = context.mark_node(std::make_shared<v1::Reshape>(src_flat, emb_table_shape, false));

    auto emb_sum = context.mark_node(std::make_shared<v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets_global));

    // emb_sum output shape: [NumOffsets, Features] = [Batch * Y, Features]
    // We need to keep only first Y-1 outputs per batch.
    // Reshape to [Batch, Y, Features]
    auto emb_sum_reshaped_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{
        batch_size_1d,
        Y_1d,
        feature_size_1d
    }, 0));
    auto emb_sum_reshaped = context.mark_node(std::make_shared<v1::Reshape>(emb_sum, emb_sum_reshaped_shape, false));

    // Slice to [Batch, Y-1, Features]
    auto slice_axes = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto slice_start = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto slice_stop = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1})); // Exclude last
    auto slice_step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));

    auto emb_sum_trimmed = context.mark_node(std::make_shared<v8::Slice>(emb_sum_reshaped, slice_start, slice_stop, slice_step, slice_axes));

    // Compute counts for mean
    // counts = indptr[:, 1:] - indptr[:, :-1]
    // indptr_flat is [Batch, Y]
    auto slice_start_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto slice_stop_max = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {max_int}));

    auto indptr_right = context.mark_node(std::make_shared<v8::Slice>(indptr_flat, slice_start_1, slice_stop_max, slice_step, slice_axes));

    // indptr_left is indptr[:, :-1]
    auto indptr_left = context.mark_node(std::make_shared<v8::Slice>(indptr_flat, slice_start, slice_stop, slice_step, slice_axes));

    auto counts = context.mark_node(std::make_shared<v1::Subtract>(indptr_right, indptr_left));
    // counts shape: [Batch, Y-1]

    // Reshape counts to [Batch, Y-1, 1] to broadcast against Features
    // We can use Reshape directly
    auto counts_reshaped_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{
        batch_size_1d,
        context.mark_node(std::make_shared<v1::Subtract>(Y_1d, one_node_1d)),
        one_node_1d
    }, 0));

    auto counts_reshaped = context.mark_node(std::make_shared<v1::Reshape>(counts, counts_reshaped_shape, false));

    auto counts_float = context.mark_node(std::make_shared<v1::ConvertLike>(counts_reshaped, src));

    auto zero_float = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto zero_like_counts = context.mark_node(std::make_shared<v1::ConvertLike>(zero_float, counts_float));
    auto one_float = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto one_like_counts = context.mark_node(std::make_shared<v1::ConvertLike>(one_float, counts_float));

    auto is_zero = context.mark_node(std::make_shared<v1::Equal>(counts_float, zero_like_counts));
    auto safe_counts = context.mark_node(std::make_shared<v1::Select>(is_zero, one_like_counts, counts_float));

    auto result_flat = context.mark_node(std::make_shared<v1::Divide>(emb_sum_trimmed, safe_counts));

    auto Y_minus_1 = context.mark_node(std::make_shared<v1::Subtract>(Y, context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}))));
    auto Y_minus_1_vec = context.mark_node(std::make_shared<v1::Reshape>(Y_minus_1, context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1})), false));

    auto output_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{src_prefix, Y_minus_1_vec, src_suffix}, 0));

    auto result = context.mark_node(std::make_shared<v1::Reshape>(result_flat, output_shape, false));

    return {result};
}

}
}
}
}
