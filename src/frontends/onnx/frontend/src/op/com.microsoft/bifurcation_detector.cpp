// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

// Translator for com.microsoft.BifurcationDetector.
//
// Reference implementation (ORT):
//   onnxruntime/contrib_ops/cpu/bert/bifurcation_detector.h
//
// Semantics:
//
//   Inputs:  src_tokens [S] int64,
//            cur_tokens [C] int64,
//            prev_suffix_match_idx (scalar) int64,
//            pred_tokens [P] int64  (optional, P == S + 1 - prev_idx)
//   Attrs:   min_ngram_size (int, default 1, >= 1)
//            max_ngram_size (int, default 1, >= min_ngram_size)
//   Outputs: tokens [variable] int64,
//            suffix_match_idx (same shape as prev_suffix_match_idx) int64
//
// Phase 1 (bifurcation merge):
//   - If pred is absent: tokens = cur_tokens.
//   - Else: scan pred_tokens vs src_tokens[prev_idx:] for first mismatch
//     at index `pred_bifur_idx in [0, S - prev_idx]`. If all pred[0..n-1]
//     match (where n = S - prev_idx), pred_bifur_idx = n. The output
//     is `cur ++ pred[0 : pred_bifur_idx + 1]`. Note the +1 means at
//     least one pred token is always appended (the bifurcating token,
//     or the trailing token when fully matching).
//
// Phase 2 (suffix match):
//   For n in [min_ngram_size, max_ngram_size]:
//     - If n > tokens_len: break.
//     - Search the suffix tokens[-n:] in src_tokens as a substring.
//       Let first_pos be the first occurrence (or absent).
//     - If absent: break (keep current suffix_idx).
//     - candidate = first_pos + n.
//     - If candidate >= src_len: suffix_idx = candidate, break.
//     - If a second occurrence exists in src[first_pos + 1 :], i.e.
//       count(matches) >= 2: suffix_idx = -1, continue.
//     - Else: suffix_idx = candidate, continue.
//   Return suffix_idx (initial value -1).
ov::OutputVector bifurcation_detector(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3, 4);

    const auto inputs = node.get_ov_inputs();
    const auto& src = inputs[0];
    const auto& cur = inputs[1];
    const auto& prev_idx = inputs[2];
    const bool has_pred = common::is_input_valid(node, 3);

    const int64_t min_ngram = node.get_attribute_value<int64_t>("min_ngram_size", 1);
    const int64_t max_ngram = node.get_attribute_value<int64_t>("max_ngram_size", 1);
    CHECK_VALID_NODE(node, min_ngram >= 1, "min_ngram_size must be >= 1");
    CHECK_VALID_NODE(node, max_ngram >= min_ngram, "max_ngram_size must be >= min_ngram_size");

    // ---- common scalar constants ----
    auto i64_scalar = [](int64_t v) {
        return v0::Constant::create(element::i64, Shape{}, {v});
    };
    auto i64_1d = [](int64_t v) {
        return v0::Constant::create(element::i64, Shape{1}, {v});
    };
    auto i64_axis0 = i64_1d(0);
    auto i64_axis0_scalar = i64_scalar(0);
    auto zero_scalar = i64_scalar(0);
    auto one_scalar = i64_scalar(1);
    auto neg_one_scalar = i64_scalar(-1);
    auto zero_1d = i64_1d(0);
    auto one_1d = i64_1d(1);

    // Tokens are int64 throughout.
    auto shape_of_i64 = [&](const ov::Output<ov::Node>& t) {
        return std::make_shared<v3::ShapeOf>(t, element::i64);
    };
    auto squeeze_to_scalar = [&](const ov::Output<ov::Node>& t) -> ov::Output<ov::Node> {
        return std::make_shared<v0::Squeeze>(t);
    };

    auto src_shape = shape_of_i64(src);
    auto src_len_scalar = squeeze_to_scalar(src_shape);

    // ============================================================
    // Phase 1: bifurcation merge.
    // ============================================================
    ov::Output<ov::Node> tokens_out;
    if (!has_pred) {
        tokens_out = cur;
    } else {
        const auto& pred = inputs[3];
        auto prev_idx_raw = squeeze_to_scalar(prev_idx);
        // Clamp prev_suffix_match_idx into [0, src_len]. A negative value (e.g. the
        // -1 sentinel) would otherwise be interpreted by v8::Slice as an index from
        // the end, producing wrong windows or out-of-bounds slicing.
        auto prev_idx_clamped_low = std::make_shared<v1::Maximum>(prev_idx_raw, zero_scalar);
        auto prev_idx_scalar = std::make_shared<v1::Minimum>(prev_idx_clamped_low, src_len_scalar);
        auto pred_shape = shape_of_i64(pred);
        auto pred_len_scalar = squeeze_to_scalar(pred_shape);
        // n = min(P, S - prev_idx); clamp to >= 0 for safety.
        // We compare only up to min(P, S - prev_idx) positions because both
        // pred[0:k] and src[prev_idx:prev_idx+k] must be in-bounds.
        auto n_raw = std::make_shared<v1::Subtract>(src_len_scalar, prev_idx_scalar);
        auto n_clipped_to_src = std::make_shared<v1::Maximum>(n_raw, zero_scalar);
        auto n_scalar = std::make_shared<v1::Minimum>(n_clipped_to_src, pred_len_scalar);

        auto n_1d = std::make_shared<v0::Unsqueeze>(n_scalar, i64_axis0);
        auto prev_idx_1d = std::make_shared<v0::Unsqueeze>(prev_idx_scalar, i64_axis0);
        // src_window_end = prev_idx + n
        auto src_window_end_scalar = std::make_shared<v1::Add>(prev_idx_scalar, n_scalar);
        auto src_window_end_1d = std::make_shared<v0::Unsqueeze>(src_window_end_scalar, i64_axis0);

        // pred_head = pred[0:n], src_window = src[prev_idx:prev_idx+n]
        auto pred_head = std::make_shared<v8::Slice>(pred, zero_1d, n_1d, one_1d, zero_1d);
        auto src_window = std::make_shared<v8::Slice>(src, prev_idx_1d, src_window_end_1d, one_1d, zero_1d);

        // First-mismatch index via cumulative-sum trick:
        //   mismatch_i64[i] = (pred[i] != src_window[i]) ? 1 : 0
        //   cum_mismatch[i] = number of mismatches in [0..i]
        //   leading_match[i] = (cum_mismatch[i] == 0)
        //   bifur = sum(leading_match)
        // For full match: cum is all-zero, leading_match all-true, bifur == n.
        // For first mismatch at k: leading_match true on [0..k-1], bifur == k.
        // For n == 0 (empty slice): ReduceSum over empty -> 0, bifur == 0.
        auto eq_bifur = std::make_shared<v1::Equal>(pred_head, src_window);
        auto mismatch_i64 = std::make_shared<v0::Convert>(std::make_shared<v1::LogicalNot>(eq_bifur), element::i64);
        auto cum_mismatch = std::make_shared<v0::CumSum>(mismatch_i64, i64_axis0_scalar);
        auto leading_match = std::make_shared<v1::Equal>(cum_mismatch, zero_scalar);
        auto bifur =
            std::make_shared<v1::ReduceSum>(std::make_shared<v0::Convert>(leading_match, element::i64), i64_axis0);

        // take = bifur + 1; pred_kept = pred[0:take]; tokens = concat(cur, pred_kept)
        auto take = std::make_shared<v1::Add>(bifur, one_scalar);
        auto take_1d = std::make_shared<v0::Unsqueeze>(take, i64_axis0);
        auto pred_kept = std::make_shared<v8::Slice>(pred, zero_1d, take_1d, one_1d, zero_1d);
        tokens_out = std::make_shared<v0::Concat>(OutputVector{cur, pred_kept}, 0);
    }

    // ============================================================
    // Phase 2: suffix match.
    // ============================================================
    auto tokens_shape = shape_of_i64(tokens_out);
    auto tokens_len_scalar = squeeze_to_scalar(tokens_shape);

    // Append a sentinel so the Gather source is never empty (src_len == 0) and any
    // clamped index stays in-bounds. src_safe = src ++ [-1], length src_len + 1.
    auto src_safe = std::make_shared<v0::Concat>(OutputVector{src, i64_1d(-1)}, 0);
    // Same sentinel trick for tokens, used by the fixed-length-n suffix gather so a
    // length-n suffix stays in-bounds even when tokens_len < n or tokens_len == 0.
    auto tokens_safe = std::make_shared<v0::Concat>(OutputVector{tokens_out, i64_1d(-1)}, 0);

    // State variables, threaded across unrolled n iterations.
    ov::Output<ov::Node> suffix_idx = neg_one_scalar;
    ov::Output<ov::Node> stopped = v0::Constant::create(element::boolean, Shape{}, {false});

    for (int64_t n = min_ngram; n <= max_ngram; ++n) {
        auto n_const = i64_scalar(n);
        // Static [0, 1, ..., n-1] offset vector. n is a compile-time constant of the
        // unrolled loop, so this (and everything keyed only on n) is a plain constant
        // rather than a dynamic Range, keeping the n-axis statically shaped.
        std::vector<int64_t> j_values(static_cast<size_t>(n));
        std::iota(j_values.begin(), j_values.end(), int64_t{0});
        auto j_const = v0::Constant::create(element::i64, Shape{static_cast<size_t>(n)}, j_values);

        // n_too_large = (n > tokens_len)
        auto n_too_large = std::make_shared<v1::Greater>(n_const, tokens_len_scalar);

        // ---- Fixed-length-n suffix via clamped gather over tokens_safe ----
        // suffix[j] = tokens_out[(tokens_len - n) + j], j in [0, n). For tokens_len >= n
        // these are the true last-n tokens; for tokens_len < n the clamped indices yield
        // a bogus suffix that is masked out by n_too_large below. Statically [n]-shaped.
        auto suffix_start = std::make_shared<v1::Subtract>(tokens_len_scalar, n_const);
        auto suffix_idx_raw = std::make_shared<v1::Add>(suffix_start, j_const);
        auto suffix_idx_low = std::make_shared<v1::Maximum>(suffix_idx_raw, zero_scalar);
        auto suffix_idx_clamped = std::make_shared<v1::Minimum>(suffix_idx_low, tokens_len_scalar);
        auto suffix_padded = std::make_shared<v8::Gather>(tokens_safe, suffix_idx_clamped, i64_axis0_scalar);
        // suffix_padded has static length n

        // ---- Build sliding windows over src: windows[i,j] = src[i+j].
        // We need num_windows >= 1 to keep the Range/Gather valid even when
        // src_len < n. The result for that degenerate case is masked out by
        // n_too_large below. The j (column) axis is the static j_const, so only the
        // num_windows (row) axis stays dynamic.
        auto num_windows_raw =
            std::make_shared<v1::Add>(std::make_shared<v1::Subtract>(src_len_scalar, n_const), one_scalar);
        auto num_windows_safe = std::make_shared<v1::Maximum>(num_windows_raw, one_scalar);
        auto i_range = std::make_shared<v4::Range>(zero_scalar, num_windows_safe, one_scalar, element::i64);
        auto i_col = std::make_shared<v0::Unsqueeze>(i_range, i64_1d(1));
        auto j_row = std::make_shared<v0::Unsqueeze>(j_const, i64_1d(0));
        auto idx = std::make_shared<v1::Add>(i_col, j_row);
        // Clamp into [0, src_len] (valid indices of src_safe). For src_len >= n the
        // real indices are all < src_len and are untouched; for shorter src the
        // out-of-range entries fold onto the sentinel and are masked out below.
        auto idx_clamped = std::make_shared<v1::Minimum>(idx, src_len_scalar);
        auto windows = std::make_shared<v8::Gather>(src_safe, idx_clamped, i64_axis0_scalar);

        // ---- Per-window match: ReduceLogicalAnd over the n-axis ----
        auto suffix_broadcast = std::make_shared<v0::Unsqueeze>(suffix_padded, i64_1d(0));
        auto eq = std::make_shared<v1::Equal>(windows, suffix_broadcast);
        auto match_per_win = std::make_shared<v1::ReduceLogicalAnd>(eq, i64_1d(1), false);
        auto match_i64 = std::make_shared<v0::Convert>(match_per_win, element::i64);
        auto count_n = std::make_shared<v1::ReduceSum>(match_i64, i64_axis0, false);

        // first_idx via cumulative-sum trick: count of leading positions with
        // cum_match == 0 equals the index of the first True in match_per_win.
        // When match_per_win is all-false, first_idx == num_windows; that result
        // is masked out by `no_match` below (skip_update covers it).
        auto cum_match = std::make_shared<v0::CumSum>(match_i64, i64_axis0_scalar);
        auto leading_nonmatch = std::make_shared<v1::Equal>(cum_match, zero_scalar);
        auto first_idx = std::make_shared<v1::ReduceSum>(std::make_shared<v0::Convert>(leading_nonmatch, element::i64),
                                                         i64_axis0,
                                                         false);

        auto candidate = std::make_shared<v1::Add>(first_idx, n_const);

        // Decision flags
        // A length-n substring cannot exist when src_len < n, so force no_match in
        // that case regardless of the (masked, sentinel-filled) window comparison.
        auto src_too_short = std::make_shared<v1::Greater>(n_const, src_len_scalar);
        auto no_match =
            std::make_shared<v1::LogicalOr>(std::make_shared<v1::Equal>(count_n, zero_scalar), src_too_short);
        auto count_ge_2 = std::make_shared<v1::Greater>(count_n, one_scalar);
        auto out_of_range = std::make_shared<v1::GreaterEqual>(candidate, src_len_scalar);

        // Updated value when we do update:
        //   if (count >= 2 AND NOT out_of_range): -1
        //   else: candidate
        auto neg1_branch = std::make_shared<v1::LogicalAnd>(count_ge_2, std::make_shared<v1::LogicalNot>(out_of_range));
        auto new_val_if_updating = std::make_shared<v1::Select>(neg1_branch, neg_one_scalar, candidate);

        // skip_update = stopped OR n_too_large OR no_match
        auto skip_update =
            std::make_shared<v1::LogicalOr>(stopped, std::make_shared<v1::LogicalOr>(n_too_large, no_match));
        auto new_suffix_idx = std::make_shared<v1::Select>(skip_update, suffix_idx, new_val_if_updating);

        // new_stopped = stopped OR n_too_large OR no_match OR out_of_range
        auto stop_now =
            std::make_shared<v1::LogicalOr>(n_too_large, std::make_shared<v1::LogicalOr>(no_match, out_of_range));
        auto new_stopped = std::make_shared<v1::LogicalOr>(stopped, stop_now);

        suffix_idx = new_suffix_idx;
        stopped = new_stopped;
    }

    // Restore suffix_match_idx output shape to match prev_suffix_match_idx.
    auto prev_idx_shape = shape_of_i64(prev_idx);
    auto suffix_idx_out = std::make_shared<v1::Reshape>(suffix_idx, prev_idx_shape, false);

    return {tokens_out, suffix_idx_out};
}

ONNX_OP("BifurcationDetector", OPSET_SINCE(1), com_microsoft::opset_1::bifurcation_detector, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
