// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer_fusion.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

using namespace ov::pass;
using namespace ov::op;

namespace {

// QSAPrepare input ports.
constexpr size_t RAW_INDEX_KEYS = 0;
constexpr size_t PAST_BLOCK_INDEX_KEYS = 1;
constexpr size_t KEY_LENGTH = 2;
constexpr size_t ROPE_COS = 3;
constexpr size_t ROPE_SIN = 4;
constexpr size_t K_NORM_WEIGHT = 5;
constexpr size_t PAST_INCOMPLETE_RAW_KEYS = 6;
constexpr size_t PAST_INCOMPLETE_ROPE = 7;

bool is_supported_data_type(const ov::element::Type& et) {
    return et == ov::element::f32 || et == ov::element::f16 || et == ov::element::bf16;
}

// An optional QSAPrepare input is disabled by an empty Constant.
bool is_disabled(const ov::Output<ov::Node>& input) {
    const auto constant = ov::as_type_ptr<v0::Constant>(input.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 0;
}

std::shared_ptr<v0::Constant> i64_const(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<v0::Constant> i64_scalar(int64_t value) {
    return v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

bool is_axes_constant(const ov::Output<ov::Node>& axes, int64_t axis, int64_t rank) {
    const auto constant = ov::as_type_ptr<v0::Constant>(axes.get_node_shared_ptr());
    if (!constant) {
        return false;
    }
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && (values[0] == axis || values[0] + rank == axis);
}

bool is_transpose_0213(const std::shared_ptr<ov::Node>& node) {
    const auto transpose = ov::as_type_ptr<v1::Transpose>(node);
    if (!transpose) {
        return false;
    }
    const auto order = ov::as_type_ptr<v0::Constant>(transpose->get_input_node_shared_ptr(1));
    return order && order->cast_vector<int64_t>() == std::vector<int64_t>{0, 2, 1, 3};
}

// Precision-only conversions (e.g. the f32 RoPE of an f16 model) do not make a table another table.
ov::Output<ov::Node> skip_converts(ov::Output<ov::Node> output) {
    while (ov::is_type<v0::Convert>(output.get_node())) {
        output = output.get_node()->input_value(0);
    }
    return output;
}

// The query RoPE tables are [N, 1, L, Dr], the QSAPrepare ones [N, L, Dr] of the very same values: the
// former has to be the latter with the head axis inserted, otherwise the fused op cannot share one table.
bool is_same_rope_table(const ov::Output<ov::Node>& query_table, const ov::Output<ov::Node>& prepare_table) {
    const auto& ps = query_table.get_partial_shape();
    if (ps.rank().is_dynamic() || ps.size() != 4 || !ps[1].compatible(1)) {
        return false;
    }
    const auto table = skip_converts(query_table);
    const auto node = table.get_node_shared_ptr();
    if (ov::is_type<v0::Unsqueeze>(node)) {
        return is_axes_constant(node->input_value(1), 1, 4) &&
               skip_converts(node->input_value(0)) == skip_converts(prepare_table);
    }
    if (ov::is_type<v1::Reshape>(node)) {
        return ps[1] == 1 && skip_converts(node->input_value(0)) == skip_converts(prepare_table);
    }
    return false;
}

bool has_batch_of(const ov::PartialShape& shape, const ov::PartialShape& raw_index_keys_shape) {
    return shape[0].is_static() && raw_index_keys_shape[0].is_static() && shape[0] == raw_index_keys_shape[0];
}

// QSAIndexer does not broadcast over the batch: a RoPE table [1 | N, L, Dr] is made [N, L, Dr].
ov::Output<ov::Node> broadcast_rope_table(const ov::Output<ov::Node>& table,
                                          const ov::Output<ov::Node>& raw_index_keys,
                                          size_t rotary_dim) {
    if (has_batch_of(table.get_partial_shape(), raw_index_keys.get_partial_shape())) {
        return table;
    }
    const auto keys_shape = std::make_shared<v3::ShapeOf>(raw_index_keys, ov::element::i64);
    const auto batch_and_length = std::make_shared<v8::Gather>(keys_shape, i64_const({0, 1}), i64_scalar(0));
    const auto target =
        std::make_shared<v0::Concat>(ov::OutputVector{batch_and_length, i64_const({static_cast<int64_t>(rotary_dim)})},
                                     0);
    return std::make_shared<v3::Broadcast>(table, target);
}

// key_length is [N] or [1] and i32 / i64, the fused op takes [N] and i32.
ov::Output<ov::Node> make_key_length(const ov::Output<ov::Node>& key_length,
                                     const ov::Output<ov::Node>& raw_index_keys) {
    ov::Output<ov::Node> result = key_length;
    if (result.get_element_type() != ov::element::i32) {
        result = std::make_shared<v0::Convert>(result, ov::element::i32);
    }
    if (!has_batch_of(key_length.get_partial_shape(), raw_index_keys.get_partial_shape())) {
        const auto keys_shape = std::make_shared<v3::ShapeOf>(raw_index_keys, ov::element::i64);
        const auto batch = std::make_shared<v8::Gather>(keys_shape, i64_const({0}), i64_scalar(0));
        result = std::make_shared<v3::Broadcast>(result, batch);
    }
    return result;
}

}  // namespace

ov::intel_cpu::QSAIndexerFusion::QSAIndexerFusion() {
    MATCHER_SCOPE(QSAIndexerFusion);

    auto rms =
        pattern::wrap_type<ov::op::internal::RMS>({pattern::any_input(pattern::rank_equals(4)), pattern::any_input()});
    // Under f16 inference the RMS keeps its f32 variance and may be followed by a conversion to the RoPE precision ...
    auto rms_out = pattern::optional<v0::Convert>({rms});
    auto rope = pattern::wrap_type<ov::op::internal::RoPE>({rms_out, pattern::any_input(), pattern::any_input()});
    // ... and, when the RoPE stays in f32, by one back to the precision of the indexer.
    auto rope_out = pattern::optional<v0::Convert>({rope});
    auto sai = pattern::wrap_type<v17::SparseAttentionIndexer>({rope_out, pattern::any_input(), pattern::any_input()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto sai_node = ov::as_type_ptr<v17::SparseAttentionIndexer>(m.get_match_root());
        const auto rope_node = ov::as_type_ptr<ov::op::internal::RoPE>(pattern_map.at(rope).get_node_shared_ptr());
        const auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms).get_node_shared_ptr());
        if (!sai_node || !rope_node || !rms_node) {
            return false;
        }
        const auto prepare_node = ov::as_type_ptr<v17::QSAPrepare>(sai_node->get_input_node_shared_ptr(1));
        if (!prepare_node || sai_node->input_value(1).get_index() != 0) {
            return false;
        }

        // ---- SparseAttentionIndexer / QSAPrepare attributes the fused op can express ----
        const int64_t key_block_size = prepare_node->get_key_block_size();
        const int64_t token_budget = sai_node->get_token_budget();
        // A single index key head cannot be split in several selection heads (H_sel is 1 or H_ik).
        if (sai_node->get_num_selection_heads() != 1 || sai_node->get_key_block_size() != key_block_size ||
            token_budget % key_block_size != 0) {
            return false;
        }
        // Both ops have to see the very same valid key lengths.
        if (sai_node->input_value(2) != prepare_node->input_value(KEY_LENGTH)) {
            return false;
        }
        // QSAIndexer always rotates and normalizes.
        for (const auto port : {ROPE_COS, ROPE_SIN, K_NORM_WEIGHT, PAST_INCOMPLETE_ROPE}) {
            if (is_disabled(prepare_node->input_value(port))) {
                return false;
            }
        }

        // ---- query RMSNorm + RoPE ----
        if (!rms_node->get_elementwise_affine() || (rms_node->get_axis() != -1 && rms_node->get_axis() != 3)) {
            return false;
        }
        const auto rope_cfg = rope_node->get_config();
        auto plain_cfg = rope_cfg;
        plain_cfg.input_trans0213 = false;
        if (!ov::op::internal::is_plain_rotate_half(plain_cfg) || rope_cfg.input_interleaved ||
            rope_cfg.use_rope_cache) {
            return false;
        }

        const auto raw_keys = prepare_node->input_value(RAW_INDEX_KEYS);
        const auto& keys_ps = raw_keys.get_partial_shape();
        const auto& cos_ps = prepare_node->get_input_partial_shape(ROPE_COS);
        const auto& sin_ps = prepare_node->get_input_partial_shape(ROPE_SIN);
        const auto& past_blocks_ps = prepare_node->get_input_partial_shape(PAST_BLOCK_INDEX_KEYS);
        if (keys_ps.rank().is_dynamic() || !keys_ps[2].is_static() || cos_ps.rank().is_dynamic() ||
            !cos_ps[2].is_static() || sin_ps.rank().is_dynamic() || !sin_ps[2].is_static() ||
            past_blocks_ps.rank().is_dynamic() ||
            prepare_node->get_input_partial_shape(KEY_LENGTH).rank().is_dynamic()) {
            return false;
        }
        const auto head_dim = static_cast<size_t>(keys_ps[2].get_length());
        const auto rotary_dim = static_cast<size_t>(cos_ps[2].get_length());
        if (rotary_dim != static_cast<size_t>(sin_ps[2].get_length()) || rotary_dim != rope_cfg.rotary_ndims) {
            return false;
        }

        // The query RoPE has to use the tables the block keys are rotated with.
        if (!is_same_rope_table(rope_node->input_value(1), prepare_node->input_value(ROPE_COS)) ||
            !is_same_rope_table(rope_node->input_value(2), prepare_node->input_value(ROPE_SIN))) {
            return false;
        }

        // Both normalizations share the epsilon of the fused op.
        if (static_cast<float>(rms_node->get_epsilon()) != prepare_node->get_epsilon()) {
            return false;
        }

        // The fused op runs the RMS and the RoPE itself on the raw [N, L, H, D] query.
        const auto rms_data = rms_node->input_value(0);
        ov::Output<ov::Node> raw_query = rms_data;
        if (!rope_cfg.input_trans0213) {
            const auto producer = rms_data.get_node_shared_ptr();
            raw_query = is_transpose_0213(producer)
                            ? producer->input_value(0)
                            : std::make_shared<v1::Transpose>(rms_data, i64_const({0, 2, 1, 3}))->output(0);
        }
        const auto& query_ps = raw_query.get_partial_shape();
        if (query_ps.rank().is_dynamic() || query_ps.size() != 4 || !query_ps[2].is_static() ||
            query_ps[3] != head_dim) {
            return false;
        }

        // q_norm_weight: [D], possibly with unit leading dimensions.
        ov::Output<ov::Node> q_norm_weight = rms_node->input_value(1);
        const auto& gamma_ps = q_norm_weight.get_partial_shape();
        if (gamma_ps.rank().is_dynamic() || gamma_ps.size() < 1 || gamma_ps[gamma_ps.size() - 1] != head_dim) {
            return false;
        }
        for (size_t i = 0; i + 1 < gamma_ps.size(); ++i) {
            if (gamma_ps[i] != 1) {
                return false;
            }
        }

        for (const auto& et : {raw_query.get_element_type(),
                               raw_keys.get_element_type(),
                               prepare_node->get_input_element_type(ROPE_COS),
                               prepare_node->get_input_element_type(ROPE_SIN),
                               q_norm_weight.get_element_type(),
                               prepare_node->get_input_element_type(K_NORM_WEIGHT),
                               prepare_node->get_input_element_type(PAST_BLOCK_INDEX_KEYS),
                               prepare_node->get_input_element_type(PAST_INCOMPLETE_RAW_KEYS),
                               prepare_node->get_input_element_type(PAST_INCOMPLETE_ROPE)}) {
            if (!is_supported_data_type(et)) {
                return false;
            }
        }

        ov::NodeVector new_nodes;
        std::shared_ptr<ov::intel_cpu::QSAIndexer> qsa_indexer;
        ov::Output<ov::Node> block_index_keys;
        ov::Output<ov::Node> incomplete_raw_keys;
        try {
            if (gamma_ps.size() != 1) {
                q_norm_weight =
                    std::make_shared<v1::Reshape>(q_norm_weight, i64_const({static_cast<int64_t>(head_dim)}), false);
            }

            // ---- fused op inputs (the QSAPrepare state tensors get the single index key head axis) ----
            const auto key_length_value = prepare_node->input_value(KEY_LENGTH);
            const auto k_idx = std::make_shared<v0::Unsqueeze>(raw_keys, i64_const({2}));
            const auto past_blocks =
                std::make_shared<v0::Unsqueeze>(prepare_node->input_value(PAST_BLOCK_INDEX_KEYS), i64_const({1}));
            const auto past_raw_keys =
                std::make_shared<v0::Unsqueeze>(prepare_node->input_value(PAST_INCOMPLETE_RAW_KEYS), i64_const({1}));

            ov::intel_cpu::QSAIndexer::Config config;
            config.compress_ratio = static_cast<size_t>(key_block_size);
            config.block_topk = static_cast<size_t>(token_budget / key_block_size);
            config.h_sel = 1;
            config.indexer_rotary_dim = rotary_dim;
            config.indexer_head_dim = head_dim;
            config.eps = prepare_node->get_epsilon();
            // SparseAttentionIndexer scores are not scaled; the scale does not change the selection anyway.
            config.scale = 1.0F;

            ov::OutputVector args(ov::intel_cpu::QSAIndexer::InputIdx::COUNT);
            using InputIdx = ov::intel_cpu::QSAIndexer::InputIdx;
            args[InputIdx::Q_IDX] = raw_query;
            args[InputIdx::K_IDX] = k_idx;
            args[InputIdx::ROPE_COS] = broadcast_rope_table(prepare_node->input_value(ROPE_COS), raw_keys, rotary_dim);
            args[InputIdx::ROPE_SIN] = broadcast_rope_table(prepare_node->input_value(ROPE_SIN), raw_keys, rotary_dim);
            args[InputIdx::KEY_LENGTH] = make_key_length(key_length_value, raw_keys);
            args[InputIdx::Q_NORM_WEIGHT] = q_norm_weight;
            args[InputIdx::K_NORM_WEIGHT] = prepare_node->input_value(K_NORM_WEIGHT);
            args[InputIdx::SUMMARY_CACHE] = past_blocks;
            args[InputIdx::PAST_INCOMPLETE_RAW_K] = past_raw_keys;
            args[InputIdx::PAST_INCOMPLETE_ROPE] = prepare_node->input_value(PAST_INCOMPLETE_ROPE);
            qsa_indexer = std::make_shared<ov::intel_cpu::QSAIndexer>(args, config);

            // ---- QSAPrepare outputs: all blocks of the longest sequence, incomplete block keys and rope ----
            using OutputIdx = ov::intel_cpu::QSAIndexer::OutputIdx;
            block_index_keys =
                std::make_shared<v0::Squeeze>(qsa_indexer->output(OutputIdx::SUMMARY_CACHE), i64_const({1}));
            incomplete_raw_keys =
                std::make_shared<v0::Squeeze>(qsa_indexer->output(OutputIdx::PAST_INCOMPLETE_RAW_K), i64_const({1}));

            new_nodes = {k_idx,
                         past_blocks,
                         past_raw_keys,
                         qsa_indexer,
                         block_index_keys.get_node_shared_ptr(),
                         incomplete_raw_keys.get_node_shared_ptr()};
        } catch (const ov::Exception&) {
            // QSAIndexer rejects the configuration: leave the subgraph as it is.
            return false;
        }

        using OutputIdx = ov::intel_cpu::QSAIndexer::OutputIdx;
        qsa_indexer->set_friendly_name(sai_node->get_friendly_name());
        ov::copy_runtime_info({sai_node, prepare_node, rope_node, rms_node}, new_nodes);
        ov::replace_node(
            sai_node,
            ov::OutputVector{qsa_indexer->output(OutputIdx::SEL_INDICES), qsa_indexer->output(OutputIdx::SEL_COUNT)});
        ov::replace_node(prepare_node,
                         ov::OutputVector{block_index_keys,
                                          incomplete_raw_keys,
                                          qsa_indexer->output(OutputIdx::PAST_INCOMPLETE_ROPE)});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sai, matcher_name);
    this->register_matcher(m, callback);
}
