// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/pattern_blocks/qsa_index_query_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::optional;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

namespace {

// QSAPrepare input ports.
constexpr size_t RAW_INDEX_KEYS = 0;
constexpr size_t KEY_LENGTH = 2;
constexpr size_t ROPE_COS = 3;
constexpr size_t ROPE_SIN = 4;
constexpr size_t K_NORM_WEIGHT = 5;
constexpr size_t PAST_INCOMPLETE_ROPE = 7;

bool is_transpose_0213_order(const ov::Output<ov::Node>& order) {
    const auto constant = ov::as_type_ptr<v0::Constant>(order.get_node_shared_ptr());
    return constant && constant->cast_vector<int64_t>() == std::vector<int64_t>{0, 2, 1, 3};
}

ov::Output<ov::Node> skip_converts(ov::Output<ov::Node> output) {
    while (ov::is_type<v0::Convert>(output.get_node())) {
        output = output.get_node()->input_value(0);
    }
    return output;
}

// An optional QSAPrepare input is disabled by an empty Constant.
bool is_disabled(const ov::Output<ov::Node>& input) {
    const auto constant = ov::as_type_ptr<v0::Constant>(input.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 0;
}

bool static_dim(const ov::PartialShape& shape, size_t rank, size_t axis, size_t& value) {
    if (shape.rank().is_dynamic() || shape.size() != rank || shape[axis].is_dynamic()) {
        return false;
    }
    value = static_cast<size_t>(shape[axis].get_length());
    return true;
}

// Precision-only conversions do not make a table another table. The query RoPE tables are [N, 1, L, Dr], the QSAPrepare
// ones [N, L, Dr] of the very same values: the former has to be the latter with the head axis inserted.
bool is_same_rope_table(const ov::Output<ov::Node>& query_table, const ov::Output<ov::Node>& prepare_table) {
    const auto& shape = query_table.get_partial_shape();
    if (shape.rank().is_dynamic() || shape.size() != 4 || !shape[1].compatible(1)) {
        return false;
    }
    const auto node = skip_converts(query_table).get_node_shared_ptr();
    if (ov::is_type<v0::Unsqueeze>(node)) {
        const auto axes = ov::util::get_constant_from_source(node->input_value(1));
        if (!axes) {
            return false;
        }
        const auto values = axes->cast_vector<int64_t>();
        return values.size() == 1 && (values[0] == 1 || values[0] == -3) &&
               skip_converts(node->input_value(0)) == skip_converts(prepare_table);
    }
    if (ov::is_type<v1::Reshape>(node)) {
        return shape[1] == 1 && skip_converts(node->input_value(0)) == skip_converts(prepare_table);
    }
    return false;
}

// The RMSNorm epsilon comes from a constant of the model type, so it is the QSAPrepare epsilon rounded to that type
// (1e-6 is not exactly representable in f16 / bf16).
bool is_same_epsilon(double rms_epsilon, const ov::element::Type& type, float prepare_epsilon) {
    float expected = prepare_epsilon;
    if (type == ov::element::f16) {
        expected = static_cast<float>(ov::float16(prepare_epsilon));
    } else if (type == ov::element::bf16) {
        expected = static_cast<float>(ov::bfloat16(prepare_epsilon));
    }
    return static_cast<float>(rms_epsilon) == expected ||
           std::abs(rms_epsilon - static_cast<double>(prepare_epsilon)) <= 1e-4 * static_cast<double>(prepare_epsilon);
}

// [start, stop) of a step-1 Slice / StridedSlice or of a VariadicSplit output along the last axis of a rank-4 tensor of
// `width` elements in it.
bool get_last_axis_slice(const ov::Output<ov::Node>& output, int64_t width, int64_t& start, int64_t& stop) {
    constexpr int64_t rank = 4;
    const auto node = output.get_node_shared_ptr();
    auto values = [&](size_t port, std::vector<int64_t>& result) {
        const auto constant = ov::util::get_constant_from_source(node->input_value(port));
        if (!constant) {
            return false;
        }
        result = constant->cast_vector<int64_t>();
        return true;
    };
    std::vector<int64_t> begin, end, step;
    if (const auto slice = ov::as_type_ptr<v8::Slice>(node)) {
        std::vector<int64_t> axes;
        if (slice->get_input_size() != 5 || !values(1, begin) || !values(2, end) || !values(3, step) ||
            !values(4, axes) || begin.size() != 1 || end.size() != 1 || step != std::vector<int64_t>{1} ||
            axes.size() != 1 || (axes[0] != rank - 1 && axes[0] != -1)) {
            return false;
        }
        start = begin[0];
        stop = end[0];
    } else if (const auto strided_slice = ov::as_type_ptr<v1::StridedSlice>(node)) {
        auto is_zero = [](const std::vector<int64_t>& mask) {
            return std::all_of(mask.begin(), mask.end(), [](int64_t bit) {
                return bit == 0;
            });
        };
        if (!is_zero(strided_slice->get_new_axis_mask()) || !is_zero(strided_slice->get_shrink_axis_mask()) ||
            !is_zero(strided_slice->get_ellipsis_mask()) || !values(1, begin) || !values(2, end) ||
            (strided_slice->get_input_size() > 3 && !values(3, step)) || begin.size() != end.size() ||
            begin.size() > static_cast<size_t>(rank)) {
            return false;
        }
        auto mask_bit = [](const std::vector<int64_t>& mask, size_t axis) {
            return axis < mask.size() && mask[axis] != 0;
        };
        const auto& begin_mask = strided_slice->get_begin_mask();
        const auto& end_mask = strided_slice->get_end_mask();
        start = 0;
        stop = std::numeric_limits<int64_t>::max();
        for (size_t axis = 0; axis < begin.size(); ++axis) {
            if (axis < step.size() && step[axis] != 1) {
                return false;
            }
            const int64_t axis_start = mask_bit(begin_mask, axis) ? 0 : begin[axis];
            const int64_t axis_stop = mask_bit(end_mask, axis) ? std::numeric_limits<int64_t>::max() : end[axis];
            if (axis + 1 == static_cast<size_t>(rank)) {
                start = axis_start;
                stop = axis_stop;
            } else if (axis_start != 0 || axis_stop < std::numeric_limits<int32_t>::max()) {
                return false;  // the other axes are taken whole
            }
        }
    } else if (const auto split = ov::as_type_ptr<v1::VariadicSplit>(node)) {
        std::vector<int64_t> axis, lengths;
        if (!values(1, axis) || !values(2, lengths) || axis.size() != 1 || (axis[0] != rank - 1 && axis[0] != -1)) {
            return false;
        }
        // A -1 length takes what the others leave.
        int64_t known = 0;
        for (const auto length : lengths) {
            known += length == -1 ? 0 : length;
        }
        for (auto& length : lengths) {
            length = length == -1 ? width - known : length;
        }
        start = 0;
        for (size_t i = 0; i < output.get_index(); ++i) {
            start += lengths[i];
        }
        stop = start + lengths[output.get_index()];
    } else {
        return false;
    }
    if (start < 0) {
        start += width;
    }
    if (stop < 0) {
        stop += width;
    }
    stop = std::min(stop, width);
    return start >= 0 && start <= stop;
}

}  // namespace

ov::pass::pattern::op::QSAIndexQueryBlock::QSAIndexQueryBlock() : Block({}, {}, "QSAIndexQueryBlock") {
    auto order_0213 = [] {
        return wrap_type<v0::Constant>(is_transpose_0213_order);
    };

    auto raw = any_input(rank_equals(4));
    auto pre_transpose = optional<v1::Transpose>({raw, order_0213()});
    auto rms_with_gain = wrap_type<ov::op::internal::RMS>({pre_transpose, any_input()});
    auto rms_without_gain = wrap_type<ov::op::internal::RMS>({pre_transpose});
    auto gain_multiply = wrap_type<v1::Multiply>({optional<v0::Convert>({rms_without_gain}), any_input()});
    // Under an f16 inference precision the RMSNorm may stay in f32 and be followed by a conversion.
    auto norm = optional<v0::Convert>({rms_with_gain | gain_multiply | rms_without_gain});
    auto mid_transpose = optional<v1::Transpose>({norm, order_0213()});
    auto x = optional<v0::Convert>({mid_transpose});

    auto cos = any_input();
    auto sin = any_input();
    auto rope = wrap_type<ov::op::internal::RoPE>({x, cos, sin});

    auto last_axis_slice = [&] {
        return wrap_type<v8::Slice>({x, any_input(), any_input(), any_input(), any_input()}) |
               wrap_type<v1::StridedSlice>({x, any_input(), any_input(), any_input()}) |
               wrap_type<v1::VariadicSplit>({x, any_input(), any_input()});
    };
    auto rotated_slice = last_axis_slice();
    auto passed_slice = last_axis_slice();
    auto partial_rope = wrap_type<ov::op::internal::RoPE>({rotated_slice, cos, sin});
    auto concat = wrap_type<v0::Concat>({optional<v0::Convert>({partial_rope}), passed_slice});

    auto rotated = optional<v0::Convert>({rope | concat});
    auto post_transpose = optional<v1::Transpose>({rotated, order_0213()});
    auto index_query = optional<v0::Convert>({post_transpose});

    m_inputs = ov::OutputVector{raw};
    m_outputs = ov::OutputVector{index_query};
    REGISTER_ANCHORS(this,
                     raw,
                     pre_transpose,
                     rms_with_gain,
                     rms_without_gain,
                     gain_multiply,
                     mid_transpose,
                     rope,
                     rotated_slice,
                     passed_slice,
                     partial_rope,
                     concat,
                     post_transpose);
}

std::optional<ov::pass::pattern::op::QSAIndexQueryBlock::Match> ov::pass::pattern::op::QSAIndexQueryBlock::get_match(
    const PatternValueMap& pattern_map,
    const std::shared_ptr<v17::SparseAttentionIndexer>& indexer) const {
    auto anchor = [&](const std::string& name) -> std::shared_ptr<ov::Node> {
        const auto value = get_anchor(name, pattern_map);
        return value ? value->get_node_shared_ptr() : nullptr;
    };

    Match match;
    match.indexer = indexer;

    // ---- the SparseAttentionIndexer / QSAPrepare pair ----
    // index_weights (DeepSeek Sparse Attention) are not part of a fused indexer, nor several selection heads.
    if (indexer->get_input_size() != 3 || indexer->get_num_selection_heads() != 1 ||
        indexer->input_value(1).get_index() != 0) {
        return std::nullopt;
    }
    match.prepare = ov::as_type_ptr<v17::QSAPrepare>(indexer->get_input_node_shared_ptr(1));
    if (!match.prepare) {
        return std::nullopt;
    }
    const auto& prepare = match.prepare;
    const int64_t key_block_size = prepare->get_key_block_size();
    const int64_t token_budget = indexer->get_token_budget();
    if (key_block_size < 2 || indexer->get_key_block_size() != key_block_size || token_budget <= 0 ||
        token_budget % key_block_size != 0) {
        return std::nullopt;
    }
    match.key_block_size = static_cast<size_t>(key_block_size);
    match.block_topk = static_cast<size_t>(token_budget / key_block_size);
    match.epsilon = prepare->get_epsilon();
    // Both ops have to see the very same valid key lengths, and a fused indexer always normalizes and rotates.
    if (indexer->input_value(2) != prepare->input_value(KEY_LENGTH)) {
        return std::nullopt;
    }
    for (const auto port : {ROPE_COS, ROPE_SIN, K_NORM_WEIGHT, PAST_INCOMPLETE_ROPE}) {
        if (is_disabled(prepare->input_value(port))) {
            return std::nullopt;
        }
    }
    size_t sin_dim = 0;
    if (!static_dim(prepare->get_input_partial_shape(RAW_INDEX_KEYS), 3, 2, match.head_dim) ||
        !static_dim(prepare->get_input_partial_shape(ROPE_COS), 3, 2, match.rotary_dim) ||
        !static_dim(prepare->get_input_partial_shape(ROPE_SIN), 3, 2, sin_dim) || sin_dim != match.rotary_dim ||
        match.rotary_dim == 0 || match.rotary_dim % 2 != 0 || match.rotary_dim > match.head_dim) {
        return std::nullopt;
    }
    size_t query_dim = 0;
    const auto& query_shape = indexer->get_input_partial_shape(0);
    if (!static_dim(query_shape, 4, 1, match.num_heads) || !static_dim(query_shape, 4, 3, query_dim) ||
        query_dim != match.head_dim) {
        return std::nullopt;
    }

    // ---- RoPE: a single op, or Concat(RoPE(x[..., :Dr]), x[..., Dr:]) ----
    const auto concat = anchor("concat");
    const auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(concat ? anchor("partial_rope") : anchor("rope"));
    if (!rope) {
        return std::nullopt;
    }
    const auto& rope_config = rope->get_config();
    auto plain_config = rope_config;
    plain_config.input_trans0213 = false;
    if (!ov::op::internal::is_plain_rotate_half(plain_config) || rope_config.input_interleaved ||
        rope_config.use_rope_cache || rope_config.rotary_ndims != match.rotary_dim) {
        return std::nullopt;
    }
    if (concat) {
        const auto width = static_cast<int64_t>(match.head_dim);
        const auto rotary_dim = static_cast<int64_t>(match.rotary_dim);
        int64_t rotated_start = 0, rotated_stop = 0, passed_start = 0, passed_stop = 0;
        // The slices are in the layout of x, the transposed input of a RoPE would not match it.
        const auto concat_axis = ov::as_type_ptr<v0::Concat>(concat)->get_axis();
        if (rope_config.input_trans0213 || concat->get_input_size() != 2 || (concat_axis != 3 && concat_axis != -1) ||
            !get_last_axis_slice(*get_anchor("rotated_slice", pattern_map), width, rotated_start, rotated_stop) ||
            !get_last_axis_slice(*get_anchor("passed_slice", pattern_map), width, passed_start, passed_stop) ||
            rotated_start != 0 || rotated_stop != rotary_dim || passed_start != rotary_dim || passed_stop != width) {
            return std::nullopt;
        }
    }
    // The query rotates with the very tables the block keys are rotated with.
    if (!is_same_rope_table(rope->input_value(1), prepare->input_value(ROPE_COS)) ||
        !is_same_rope_table(rope->input_value(2), prepare->input_value(ROPE_SIN))) {
        return std::nullopt;
    }

    // ---- RMSNorm: RMS(x, gain), RMS(x) * gain or RMS(x) with an all-ones gain ----
    const auto rms = ov::as_type_ptr<ov::op::internal::RMS>(anchor("rms_with_gain") ? anchor("rms_with_gain")
                                                                                    : anchor("rms_without_gain"));
    if (!rms || (rms->get_input_size() == 2) != rms->get_elementwise_affine() ||
        (rms->get_axis() != -1 && rms->get_axis() != 3) ||
        !is_same_epsilon(rms->get_epsilon(), rms->get_output_element_type(0), match.epsilon)) {
        return std::nullopt;
    }
    if (rms->get_input_size() == 2) {
        match.q_norm_weight = rms->input_value(1);
    } else if (const auto gain_multiply = anchor("gain_multiply")) {
        const auto rms_side = skip_converts(gain_multiply->input_value(0)).get_node() == rms.get() ? 0 : 1;
        match.q_norm_weight = gain_multiply->input_value(1 - rms_side);
    }
    if (match.q_norm_weight.get_node()) {
        const auto& gain_shape = match.q_norm_weight.get_partial_shape();
        if (gain_shape.rank().is_dynamic() || gain_shape.size() < 1 ||
            gain_shape[gain_shape.size() - 1] != static_cast<int64_t>(match.head_dim)) {
            return std::nullopt;
        }
        for (size_t axis = 0; axis + 1 < gain_shape.size(); ++axis) {
            if (gain_shape[axis] != 1) {
                return std::nullopt;
            }
        }
    }

    // ---- layout: every transpose (an absorbed one included) swaps the length and head axes ----
    size_t swaps = rope_config.input_trans0213 ? 1 : 0;
    for (const auto* name : {"pre_transpose", "mid_transpose", "post_transpose"}) {
        swaps += anchor(name) ? 1 : 0;
    }
    match.raw_query = *get_anchor("raw", pattern_map);
    match.raw_query_blhd = swaps % 2 == 1;
    size_t raw_heads = 0, raw_dim = 0;
    if (!static_dim(match.raw_query.get_partial_shape(), 4, match.raw_query_blhd ? 2 : 1, raw_heads) ||
        !static_dim(match.raw_query.get_partial_shape(), 4, 3, raw_dim) || raw_heads != match.num_heads ||
        raw_dim != match.head_dim) {
        return std::nullopt;
    }

    for (const auto* name : {"pre_transpose",
                             "rms_with_gain",
                             "rms_without_gain",
                             "gain_multiply",
                             "mid_transpose",
                             "rope",
                             "rotated_slice",
                             "passed_slice",
                             "partial_rope",
                             "concat",
                             "post_transpose"}) {
        if (const auto node = anchor(name)) {
            match.nodes.push_back(node);
        }
    }
    return match;
}
