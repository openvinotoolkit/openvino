// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/experimental/draft_block_mask.hpp"
#include "utils.hpp"

namespace ov::op::experimental {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const DraftBlockMask* op, const std::vector<T>& input_shapes) {
    using DimType = typename T::value_type;
    const auto input_size = input_shapes.size();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_size == 3 || input_size == 5,
                           "DraftBlockMask expects 3 or 5 inputs.");

    const auto& q_shape = input_shapes[0];
    const auto& k_shape = input_shapes[1];
    const auto& mask_shape = input_shapes[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           q_shape.rank().compatible(3) && k_shape.rank().compatible(3),
                           "q_windows and k_windows must be rank 3: [N, block_s, D].");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, mask_shape.rank().compatible(2), "local_mask must be rank 2: [HW, HW].");
    if (input_size == 5) {
        for (const size_t idx : {size_t{3}, size_t{4}}) {
            const auto& r = input_shapes[idx].rank();
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   r.is_dynamic() || r.get_length() == 0 || r.get_length() == 1,
                                   "seqlen/topk inputs must be rank 0 or 1.");
        }
    }

    const auto num_heads = op->get_num_heads();
    const auto batch_size = op->get_batch_size();

    DimType Nq = q_shape.rank().is_static() ? q_shape[0] : DimType();
    DimType Nk = k_shape.rank().is_static() ? k_shape[0] : DimType();
    const DimType D = q_shape.rank().is_static() ? q_shape[2] : DimType();

    if (D.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               D.get_length() % num_heads == 0,
                               "q_windows/k_windows feature dim (",
                               D,
                               ") must be divisible by num_heads (",
                               num_heads,
                               ").");
    }

    // local_mask is always square: [HW, HW]. HW is local_mask's own extent -- the single source of
    // truth, never derived from Nq/Nk/seqlen. This check needs no seqlen VALUE, so it runs
    // unconditionally (both 3- and 5-input cases).
    DimType HW;
    if (mask_shape.rank().is_static()) {
        HW = mask_shape[0];
        if (mask_shape[0].is_static() && mask_shape[1].is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   mask_shape[0].get_length() == mask_shape[1].get_length(),
                                   "local_mask must be square: [HW, HW], got ",
                                   mask_shape,
                                   ".");
        }
    }

    // Nk's own temporal group count (Nk / HW) is independent of seqlen in general (e.g. a streaming
    // decoder's accumulated key/value cache growing past the current query window), so this check
    // needs no seqlen VALUE either -- it runs unconditionally (both 3- and 5-input cases).
    if (HW.is_static() && HW.get_length() > 0 && Nk.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               Nk.get_length() % HW.get_length() == 0,
                               "k_windows block count (",
                               Nk,
                               ") must be divisible by local_mask's HW (",
                               HW,
                               ").");
    }

    // `seqlen` is a compile-time attribute only in the 3-input case -- the Nq == seqlen * HW
    // cross-check below needs its VALUE, which does not exist when it is instead a runtime input
    // (5-input case). This mirrors BlockSparseAttention's shape_infer having no analogous
    // cross-check on its optional runtime `scale` input: the 5-input path simply narrows the output
    // shape from whatever Nq/Nk/num_heads/batch_size already give it, deferring seqlen/topk-dependent
    // validation to the CM implementation (see draft_block_mask.cpp under intel_gpu/src/graph/impls/
    // cm/ resolving seqlen from local_mask's own shape at execution time) and to
    // DraftBlockMask::evaluate() for the CPU path.
    if (input_size == 3) {
        const auto seqlen = op->get_seqlen();
        if (Nq.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   Nq.get_length() % seqlen == 0,
                                   "q_windows block count (",
                                   Nq,
                                   ") must be divisible by seqlen (",
                                   seqlen,
                                   ").");
        }
        if (HW.is_static() && Nq.is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   Nq.get_length() == seqlen * HW.get_length(),
                                   "q_windows block count (",
                                   Nq,
                                   ") must equal seqlen (",
                                   seqlen,
                                   ") * local_mask's HW (",
                                   HW,
                                   ").");
        }
    }

    auto output_shape = TRShape{DimType(batch_size), DimType(num_heads), Nq, Nk};
    return {output_shape};
}

}  // namespace ov::op::experimental