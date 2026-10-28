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
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
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
#include "openvino/pass/pattern/op/or.hpp"
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

// The RMS epsilon comes from a constant of the model type, so it is the QSAPrepare epsilon rounded to that type
// (1e-6 is not exactly representable in f16 / bf16).
bool is_same_epsilon(double rms_epsilon, const ov::element::Type& type, float prepare_epsilon) {
    float expected = prepare_epsilon;
    if (type == ov::element::f16) {
        expected = static_cast<float>(ov::float16(prepare_epsilon));
    } else if (type == ov::element::bf16) {
        expected = static_cast<float>(ov::bfloat16(prepare_epsilon));
    }
    return static_cast<float>(rms_epsilon) == expected;
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

    // The RMS has a gamma input, or none when the gamma is all ones (e.g. a zero `w` of a `1 + w` norm, whose unit
    // Multiply is folded away before RMSFusion).
    auto rms_with_gamma =
        pattern::wrap_type<ov::op::internal::RMS>({pattern::any_input(pattern::rank_equals(4)), pattern::any_input()});
    auto rms_without_gamma = pattern::wrap_type<ov::op::internal::RMS>({pattern::any_input(pattern::rank_equals(4))});
    auto rms = std::make_shared<pattern::op::Or>(ov::OutputVector{rms_with_gamma, rms_without_gamma});
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
        const auto& rms_value =
            pattern_map.count(rms_with_gamma) ? pattern_map.at(rms_with_gamma) : pattern_map.at(rms_without_gamma);
        const auto rms_node = ov::as_type_ptr<ov::op::internal::RMS>(rms_value.get_node_shared_ptr());
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
        const bool rms_has_gamma = rms_node->get_input_size() == 2;
        if (rms_has_gamma != rms_node->get_elementwise_affine() ||
            (rms_node->get_axis() != -1 && rms_node->get_axis() != 3)) {
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
        if (!is_same_epsilon(rms_node->get_epsilon(),
                             rms_node->get_output_element_type(0),
                             prepare_node->get_epsilon())) {
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

        // q_norm_weight: [D], possibly with unit leading dimensions; ones for an RMS without gamma.
        ov::Output<ov::Node> q_norm_weight =
            rms_has_gamma ? rms_node->input_value(1)
                          : v0::Constant::create(rms_node->get_output_element_type(0), ov::Shape{head_dim}, {1.0F})
                                ->output(0);
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

        // The fused op needs one precision for all its floating-point ports, but the surrounding graph can mix
        // them (e.g. an f16 model keeps the query RMS path and its weight in f32). Everything is brought to the
        // precision of the big persistent state, the summary cache, and the state outputs are brought back.
        // An existing Convert is looked through, a Constant is converted in place, a new Convert is the last resort.
        const auto data_type = prepare_node->get_input_element_type(PAST_BLOCK_INDEX_KEYS);
        if (!is_supported_data_type(data_type)) {
            return false;
        }
        ov::NodeVector new_nodes;
        auto to_type = [&new_nodes](const ov::Output<ov::Node>& output,
                                    const ov::element::Type& type) -> ov::Output<ov::Node> {
            if (output.get_element_type() == type) {
                return output;
            }
            auto source = output;
            if (ov::is_type<v0::Convert>(source.get_node())) {
                source = source.get_node()->input_value(0);
                if (source.get_element_type() == type) {
                    return source;
                }
            }
            const auto convert = std::make_shared<v0::Convert>(source, type);
            if (const auto constant = ov::util::get_constant_from_source(convert)) {
                new_nodes.push_back(constant);
                return constant;
            }
            new_nodes.push_back(convert);
            return convert;
        };
        std::shared_ptr<ov::intel_cpu::QSAIndexer> qsa_indexer;
        ov::Output<ov::Node> block_index_keys;
        ov::Output<ov::Node> incomplete_raw_keys;
        ov::Output<ov::Node> incomplete_rope;
        try {
            if (gamma_ps.size() != 1) {
                q_norm_weight =
                    std::make_shared<v1::Reshape>(q_norm_weight, i64_const({static_cast<int64_t>(head_dim)}), false);
            }
            q_norm_weight = to_type(q_norm_weight, data_type);

            // ---- fused op inputs (the QSAPrepare state tensors get the single index key head axis) ----
            const auto key_length_value = prepare_node->input_value(KEY_LENGTH);
            const auto k_idx = std::make_shared<v0::Unsqueeze>(to_type(raw_keys, data_type), i64_const({2}));
            const auto past_blocks =
                std::make_shared<v0::Unsqueeze>(prepare_node->input_value(PAST_BLOCK_INDEX_KEYS), i64_const({1}));
            const auto past_raw_keys =
                std::make_shared<v0::Unsqueeze>(to_type(prepare_node->input_value(PAST_INCOMPLETE_RAW_KEYS), data_type),
                                                i64_const({1}));

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
            args[InputIdx::Q_IDX] = to_type(raw_query, data_type);
            args[InputIdx::K_IDX] = k_idx;
            args[InputIdx::ROPE_COS] =
                broadcast_rope_table(to_type(prepare_node->input_value(ROPE_COS), data_type), raw_keys, rotary_dim);
            args[InputIdx::ROPE_SIN] =
                broadcast_rope_table(to_type(prepare_node->input_value(ROPE_SIN), data_type), raw_keys, rotary_dim);
            args[InputIdx::KEY_LENGTH] = make_key_length(key_length_value, raw_keys);
            args[InputIdx::Q_NORM_WEIGHT] = q_norm_weight;
            args[InputIdx::K_NORM_WEIGHT] = to_type(prepare_node->input_value(K_NORM_WEIGHT), data_type);
            args[InputIdx::SUMMARY_CACHE] = past_blocks;
            args[InputIdx::PAST_INCOMPLETE_RAW_K] = past_raw_keys;
            args[InputIdx::PAST_INCOMPLETE_ROPE] = to_type(prepare_node->input_value(PAST_INCOMPLETE_ROPE), data_type);
            qsa_indexer = std::make_shared<ov::intel_cpu::QSAIndexer>(args, config);

            // ---- QSAPrepare outputs: all blocks of the longest sequence, incomplete block keys and rope ----
            using OutputIdx = ov::intel_cpu::QSAIndexer::OutputIdx;
            block_index_keys =
                to_type(std::make_shared<v0::Squeeze>(qsa_indexer->output(OutputIdx::SUMMARY_CACHE), i64_const({1})),
                        prepare_node->get_output_element_type(0));
            incomplete_raw_keys = to_type(
                std::make_shared<v0::Squeeze>(qsa_indexer->output(OutputIdx::PAST_INCOMPLETE_RAW_K), i64_const({1})),
                prepare_node->get_output_element_type(1));
            incomplete_rope =
                to_type(qsa_indexer->output(OutputIdx::PAST_INCOMPLETE_ROPE), prepare_node->get_output_element_type(2));

            for (const auto& node : {std::shared_ptr<ov::Node>(k_idx),
                                     std::shared_ptr<ov::Node>(past_blocks),
                                     std::shared_ptr<ov::Node>(past_raw_keys),
                                     std::shared_ptr<ov::Node>(qsa_indexer),
                                     block_index_keys.get_node_shared_ptr(),
                                     incomplete_raw_keys.get_node_shared_ptr()}) {
                new_nodes.push_back(node);
            }
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
        ov::replace_node(prepare_node, ov::OutputVector{block_index_keys, incomplete_raw_keys, incomplete_rope});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sai, matcher_name);
    this->register_matcher(m, callback);
}
