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
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "transformations/pattern_blocks/qsa_index_query_block.hpp"

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

std::shared_ptr<v0::Constant> i64_const(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// key_length is [N] or [1] and i32 / i64, the fused op takes [N | 1] and i32.
ov::Output<ov::Node> make_key_length(const ov::Output<ov::Node>& key_length) {
    if (key_length.get_element_type() == ov::element::i32) {
        return key_length;
    }
    return std::make_shared<v0::Convert>(key_length, ov::element::i32);
}

}  // namespace

ov::intel_cpu::QSAIndexerFusion::QSAIndexerFusion() {
    MATCHER_SCOPE(QSAIndexerFusion);

    // The index query RMSNorm + RoPE, which QSAIndexer applies itself to the raw query.
    auto index_query = std::make_shared<pattern::op::QSAIndexQueryBlock>();
    auto sai = pattern::wrap_type<v17::SparseAttentionIndexer>(
        {index_query, pattern::wrap_type<v17::QSAPrepare>(), pattern::any_input()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto sai_node = ov::as_type_ptr<v17::SparseAttentionIndexer>(m.get_match_root());
        const auto match = index_query->get_match(m.get_pattern_value_map(), sai_node);
        if (!match) {
            return false;
        }
        const auto& prepare_node = match->prepare;
        const auto head_dim = match->head_dim;
        const auto rotary_dim = match->rotary_dim;
        const auto raw_keys = prepare_node->input_value(RAW_INDEX_KEYS);
        if (prepare_node->get_input_partial_shape(PAST_BLOCK_INDEX_KEYS).rank().is_dynamic() ||
            prepare_node->get_input_partial_shape(KEY_LENGTH).rank().is_dynamic()) {
            return false;
        }

        // The fused op runs the RMS and the RoPE itself on the raw [N, L, H, D] query.
        ov::NodeVector new_nodes;
        ov::Output<ov::Node> raw_query = match->raw_query;
        if (!match->raw_query_blhd) {
            raw_query = std::make_shared<v1::Transpose>(raw_query, i64_const({0, 2, 1, 3}));
            new_nodes.push_back(raw_query.get_node_shared_ptr());
        }

        // q_norm_weight: [D], possibly with unit leading dimensions; ones for an RMS without gain.
        const auto data_type = prepare_node->get_input_element_type(PAST_BLOCK_INDEX_KEYS);
        ov::Output<ov::Node> q_norm_weight =
            match->q_norm_weight.get_node() ? match->q_norm_weight
                                            : v0::Constant::create(data_type, ov::Shape{head_dim}, {1.0F})->output(0);
        const auto& gamma_ps = q_norm_weight.get_partial_shape();

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
        if (!is_supported_data_type(data_type)) {
            return false;
        }
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
            config.compress_ratio = match->key_block_size;
            config.block_topk = match->block_topk;
            config.h_sel = 1;
            config.indexer_rotary_dim = rotary_dim;
            config.indexer_head_dim = head_dim;
            config.eps = match->epsilon;
            // SparseAttentionIndexer scores are not scaled; the scale does not change the selection anyway.
            config.scale = 1.0F;

            ov::OutputVector args(ov::intel_cpu::QSAIndexer::InputIdx::COUNT);
            using InputIdx = ov::intel_cpu::QSAIndexer::InputIdx;
            args[InputIdx::Q_IDX] = to_type(raw_query, data_type);
            args[InputIdx::K_IDX] = k_idx;
            // the RoPE tables [1 | N, L, Dr] are shared by the batches, QSAIndexer takes them as they are
            args[InputIdx::ROPE_COS] = to_type(prepare_node->input_value(ROPE_COS), data_type);
            args[InputIdx::ROPE_SIN] = to_type(prepare_node->input_value(ROPE_SIN), data_type);
            args[InputIdx::KEY_LENGTH] = make_key_length(key_length_value);
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
        auto matched_nodes = match->nodes;
        matched_nodes.push_back(sai_node);
        matched_nodes.push_back(prepare_node);
        ov::copy_runtime_info(matched_nodes, new_nodes);
        ov::replace_node(
            sai_node,
            ov::OutputVector{qsa_indexer->output(OutputIdx::SEL_INDICES), qsa_indexer->output(OutputIdx::SEL_COUNT)});
        ov::replace_node(prepare_node, ov::OutputVector{block_index_keys, incomplete_raw_keys, incomplete_rope});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(sai, matcher_name);
    this->register_matcher(m, callback);
}
