// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_segment_mean_csr(const NodeContext& context) {
    // torch_scatter::segment_mean_csr(Tensor src, Tensor indptr, Tensor? out=None) -> Tensor
    //
    // Performs segmented mean reduction using CSR (Compressed Sparse Row) format.
    // The segmentation dimension is determined by indptr.dim() - 1.
    //
    // Example:
    //   src = torch.randn(10, 6, 64)
    //   indptr = torch.tensor([[0, 2, 5, 6]])  # shape (1, 4), so axis = 2 - 1 = 1
    //   result = segment_csr(src, indptr, reduce="mean")
    //   # Segment 0: mean of src[:, 0:2, :]
    //   # Segment 1: mean of src[:, 2:5, :]
    //   # Segment 2: mean of src[:, 5:6, :]
    //   # Output shape: (10, 3, 64)
    //
    // Algorithm: Uses prefix sums for efficient segment reduction without unrolling.
    //   1. Compute cumulative sum along the segmentation axis
    //   2. Gather boundary values using indptr
    //   3. Compute segment sums as difference of consecutive boundaries
    //   4. Divide by segment lengths to get means
    //
    // Limitations:
    //   - indptr and src ranks must be static
    //   - Empty segments (indptr[i+1] == indptr[i]) produce 0 (length clamped to 1)

    num_inputs_check(context, 2, 3);
    auto src = context.get_input(0);
    auto indptr = context.get_input(1);

    // Validate indptr shape
    auto indptr_pshape = indptr.get_partial_shape();
    PYTORCH_OP_CONVERSION_CHECK(indptr_pshape.rank().is_static(),
                                "segment_mean_csr: indptr rank must be static");
    auto indptr_rank = indptr_pshape.rank().get_length();
    PYTORCH_OP_CONVERSION_CHECK(indptr_rank >= 1, "segment_mean_csr: indptr rank must be >= 1");

    // Validate src shape
    auto src_pshape = src.get_partial_shape();
    PYTORCH_OP_CONVERSION_CHECK(src_pshape.rank().is_static(), "segment_mean_csr: src rank must be static");
    auto src_rank = src_pshape.rank().get_length();

    // Segmentation axis = indptr.dim() - 1 (per torch_scatter semantics)
    int64_t axis = static_cast<int64_t>(indptr_rank) - 1;
    PYTORCH_OP_CONVERSION_CHECK(axis >= 0 && axis < static_cast<int64_t>(src_rank),
                                "segment_mean_csr: axis ",
                                axis,
                                " is out of range for src rank ",
                                src_rank);

    // Create constants - use i64 for indices to avoid overflow with large sparse structures
    auto axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {axis}));
    auto axis_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {axis}));
    auto axis_0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto axis_0_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto zero_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto zero_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));

    // Convert indptr to i64 to support large indices (Gather supports i32 or i64)
    auto indptr_i64 = context.mark_node(std::make_shared<v0::Convert>(indptr, element::i64));
    Output<Node> indptr_indices = indptr_i64;

    // Handle broadcasting for batch dimensions (when axis > 0)
    if (axis > 0) {
        auto src_shape = context.mark_node(std::make_shared<v3::ShapeOf>(src, element::i64));
        auto indptr_shape = context.mark_node(std::make_shared<v3::ShapeOf>(indptr_i64, element::i64));
        auto batch_shape =
            context.mark_node(std::make_shared<v8::Slice>(src_shape, zero_1d, axis_1d, one_1d, axis_0_1d));
        auto indptr_last_dim = context.mark_node(std::make_shared<v8::Gather>(indptr_shape, axis_1d, axis_0));
        auto target_shape =
            context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_shape, indptr_last_dim}, 0));
        indptr_indices = context.mark_node(
            std::make_shared<v3::Broadcast>(indptr_i64, target_shape, BroadcastType::BIDIRECTIONAL));
    }

    // Create zero-padded source for prefix sum computation
    // We need a leading zero slice along the axis so boundaries[0] = 0
    auto zero_slice = context.mark_node(std::make_shared<v8::Gather>(src, zero_1d, axis_const));
    auto zero_like = context.mark_node(std::make_shared<v1::ConvertLike>(zero_scalar, src));
    auto zero_padded = context.mark_node(std::make_shared<v1::Multiply>(zero_slice, zero_like));
    auto src_padded = context.mark_node(std::make_shared<v0::Concat>(OutputVector{zero_padded, src}, axis));

    // Compute prefix sums along the segmentation axis
    auto prefix = context.mark_node(std::make_shared<v0::CumSum>(src_padded, axis_const));

    // Gather boundary values at indptr positions
    auto boundaries =
        context.mark_node(std::make_shared<v8::Gather>(prefix, indptr_indices, axis_const, axis));

    // Compute segment lengths and sums from boundaries
    auto indptr_shape = context.mark_node(std::make_shared<v3::ShapeOf>(indptr_indices, element::i64));
    auto axis_len = context.mark_node(std::make_shared<v8::Gather>(indptr_shape, axis_1d, axis_0));
    auto end_left = context.mark_node(std::make_shared<v1::Subtract>(axis_len, one_1d));

    // indptr_right = indptr[1:], indptr_left = indptr[:-1]
    auto indptr_right =
        context.mark_node(std::make_shared<v8::Slice>(indptr_indices, one_1d, axis_len, one_1d, axis_1d));
    auto indptr_left =
        context.mark_node(std::make_shared<v8::Slice>(indptr_indices, zero_1d, end_left, one_1d, axis_1d));
    auto lens = context.mark_node(std::make_shared<v1::Subtract>(indptr_right, indptr_left));

    // boundaries_right = boundaries[1:], boundaries_left = boundaries[:-1]
    auto boundaries_right =
        context.mark_node(std::make_shared<v8::Slice>(boundaries, one_1d, axis_len, one_1d, axis_1d));
    auto boundaries_left =
        context.mark_node(std::make_shared<v8::Slice>(boundaries, zero_1d, end_left, one_1d, axis_1d));
    auto sums = context.mark_node(std::make_shared<v1::Subtract>(boundaries_right, boundaries_left));

    // Handle empty segments: clamp length to 1 to avoid division by zero
    // Empty segments will produce 0 (sum=0, len=1 -> mean=0)
    auto lens_nonzero = context.mark_node(std::make_shared<v1::Maximum>(lens, one_1d));

    // Reshape lens to broadcast correctly with sums for trailing dimensions
    Output<Node> lens_reshaped = lens_nonzero;
    auto tail_rank = static_cast<int64_t>(src_rank) - axis - 1;
    if (tail_rank > 0) {
        std::vector<int64_t> ones_vals(static_cast<size_t>(tail_rank), 1);
        auto ones_tail = context.mark_node(
            v0::Constant::create(element::i64, Shape{static_cast<size_t>(tail_rank)}, ones_vals));
        auto lens_shape = context.mark_node(std::make_shared<v3::ShapeOf>(lens_nonzero, element::i64));
        auto new_lens_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{lens_shape, ones_tail}, 0));
        lens_reshaped = context.mark_node(std::make_shared<v1::Reshape>(lens_nonzero, new_lens_shape, false));
    }

    // Compute mean = sums / lens
    auto lens_cast = context.mark_node(std::make_shared<v1::ConvertLike>(lens_reshaped, sums));
    auto mean = context.mark_node(std::make_shared<v1::Divide>(sums, lens_cast));

    // Handle optional output tensor (guard against out-of-range access)
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        context.mutate_input(2, mean);
    }
    return {context.mark_output(mean)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov

