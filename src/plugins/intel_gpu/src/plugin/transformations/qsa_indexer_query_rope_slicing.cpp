// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer_query_rope_slicing.hpp"

#include <memory>
#include <optional>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

std::optional<std::vector<int64_t>> constant_values(const ov::Output<ov::Node>& output) {
    if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr())) {
        return constant->cast_vector<int64_t>();
    }
    return std::nullopt;
}

// Whether `output` only feeds the query of a QSA indexer, possibly through Transpose / Reshape nodes.
bool only_feeds_indexer_query(ov::Output<ov::Node> output) {
    while (true) {
        const auto targets = output.get_target_inputs();
        if (targets.size() != 1) {
            return false;
        }
        const auto& target = *targets.begin();
        auto* node = target.get_node();
        if (ov::is_type_any_of<ov::op::v17::SparseAttentionIndexer, ov::op::internal::QSAIndexer, ov::op::internal::PagedQSAIndexer>(node)) {
            return target.get_index() == 0;
        }
        if (!ov::is_type_any_of<ov::op::v1::Transpose, ov::op::v1::Reshape>(node)) {
            return false;
        }
        output = node->output(0);
    }
}

// The cos / sin rows [B, L, rotary] with a head axis, as an Unsqueeze(axis = 1): NopElimination may have turned it
// into a Reshape, which FuseQSAIndexerQueryRoPE only takes for a static length.
ov::Output<ov::Node> with_head_axis(const ov::Output<ov::Node>& rows) {
    const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(rows.get_node_shared_ptr());
    if (!reshape) {
        return rows;
    }
    const auto& in = reshape->get_input_partial_shape(0);
    if (in.rank() != 3 || rows.get_partial_shape() != ov::PartialShape{in[0], 1, in[1], in[2]}) {
        return rows;
    }
    return std::make_shared<ov::op::v0::Unsqueeze>(reshape->input_value(0), ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
}

}  // namespace

QSAIndexerQueryRoPESlicing::QSAIndexerQueryRoPESlicing() {
    using namespace ov::pass::pattern;
    using RoPE = ov::op::internal::RoPE;

    // The head-first query [B, H, L, D], split along the head size into {rotary, rest}.
    const auto p_head_first = any_input(rank_equals(4));
    const auto p_split =
        wrap_type_strict_index<ov::op::v1::VariadicSplit>({p_head_first, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    const auto p_rope = wrap_type<RoPE>({p_split->output(0), any_input(), any_input()});
    const auto p_concat = wrap_type<ov::op::v0::Concat>({p_rope, p_split->output(1)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        const auto rope = ov::as_type_ptr<RoPE>(pattern_map.at(p_rope).get_node_shared_ptr());
        const auto split = pattern_map.at(p_split).get_node_shared_ptr();
        if (!concat || !rope || transformation_callback(concat) || !only_feeds_indexer_query(concat->output(0))) {
            return false;
        }

        const auto& head_first = pattern_map.at(p_head_first);
        const auto head_size = head_first.get_partial_shape()[3];
        const auto split_axis = constant_values(split->input_value(1));
        const auto lengths = constant_values(split->input_value(2));
        if (head_size.is_dynamic() || !split_axis || split_axis->size() != 1 || ov::util::normalize((*split_axis)[0], 4) != 3 || !lengths ||
            lengths->size() != 2 || ov::util::normalize(concat->get_axis(), 4) != 3) {
            return false;
        }

        // A plain rotate-half RoPE over the whole rotary part, the rest passed through.
        auto config = rope->get_config();
        const auto rotary = static_cast<int64_t>(config.rotary_ndims);
        const auto& cos_shape = rope->get_input_partial_shape(1);
        if (!ov::op::internal::is_plain_rotate_half(config) || config.input_interleaved || config.use_rope_cache || cos_shape.rank().is_dynamic() ||
            cos_shape[-1] != rotary || (*lengths)[0] != rotary || ((*lengths)[1] != head_size.get_length() - rotary && (*lengths)[1] != -1) ||
            rotary >= head_size.get_length()) {
            return false;
        }

        // Fold a Transpose(0, 2, 1, 3) of the token-major query into the RoPE (input_trans0213), the form
        // FuseQSAIndexerQueryRoPE expects before the token-major Transpose of the dense indexer. Otherwise (e.g. the
        // single-token paged query, whose Transpose became a Reshape) the RoPE takes the head-first query as it is.
        ov::NodeVector sources{split, rope, concat};
        ov::Output<ov::Node> input = head_first;
        if (const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(head_first.get_node_shared_ptr())) {
            const auto order = constant_values(transpose->input_value(1));
            if (order && *order == std::vector<int64_t>{0, 2, 1, 3}) {
                input = transpose->input_value(0);
                config.input_trans0213 = true;
                sources.push_back(transpose);
            }
        }
        const auto sliced = std::make_shared<RoPE>(ov::OutputVector{input, with_head_axis(rope->input_value(1)), with_head_axis(rope->input_value(2))}, config);
        sliced->set_friendly_name(rope->get_friendly_name());
        ov::copy_runtime_info(sources, sliced);
        ov::replace_node(concat, sliced);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_concat, "QSAIndexerQueryRoPESlicing"), callback);
}

}  // namespace ov::intel_gpu
