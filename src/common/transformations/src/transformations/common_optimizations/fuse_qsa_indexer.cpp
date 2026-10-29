// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <cmath>
#include <string>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/rms_norm.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using ov::op::internal::QSAIndexer;
using ov::op::v17::QSAPrepare;
using ov::op::v17::SparseAttentionIndexer;

bool is_supported_epsilon(float epsilon) {
    return std::isfinite(epsilon) && epsilon >= QSAIndexer::min_epsilon && epsilon <= QSAIndexer::max_epsilon;
}

bool is_supported_epsilon_constant(const ov::Output<ov::Node>& output) {
    float value = 0.0f;
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return ov::op::util::get_single_value(constant, value) && is_supported_epsilon(value);
}

// QSAIndexer replaces every QSAPrepare output, so each of them must be consumed only by the fused subgraph.
bool all_outputs_have_single_consumer(const ov::Output<ov::Node>& output) {
    const auto consumers_count = ov::pass::pattern::consumers_count(1);
    for (const auto& node_output : output.get_node()->outputs()) {
        if (!consumers_count(node_output)) {
            return false;
        }
    }
    return true;
}

}  // namespace

ov::pass::FuseQSAIndexer::FuseQSAIndexer() {
    MATCHER_SCOPE(FuseQSAIndexer);
    using namespace ov::pass::pattern;

    // Query RMSNorm: internal RMSNorm op or its decomposition x / Sqrt(ReduceMean(x^2) + eps) * gamma.
    // Both alternatives bind the same `query_m` / `query_weight_m` labels.
    // An empty Constant disables the optional normalization weight; QSAIndexer requires both norms enabled.
    const auto enabled_norm_weight = [](const ov::Output<ov::Node>& output) {
        return !ov::util::is_empty_constant_tensor(output);
    };
    const auto query_m = any_input(shape_matches("?, " + std::to_string(QSAIndexer::supported_num_heads) + ", ?, " +
                                                 std::to_string(QSAIndexer::supported_head_size)));
    const auto query_weight_m = any_input(enabled_norm_weight);
    const auto last_axis = value_matches("-1") || value_matches("3");

    const auto rms_norm_m = wrap_type<ov::op::internal::RMSNorm>(
        {query_m, wrap_type<ov::op::v0::Constant>(last_axis), query_weight_m},
        consumers_count(1) && [](const ov::Output<ov::Node>& output) {
            const auto epsilon = ov::as_type<ov::op::internal::RMSNorm>(output.get_node())->get_epsilon();
            return is_supported_epsilon(static_cast<float>(epsilon));
        });

    const auto power_m = wrap_type<ov::op::v1::Power>({query_m, wrap_type<ov::op::v0::Constant>(value_matches("2"))},
                                                      consumers_count(1));
    const auto mean_m = wrap_type<ov::op::v1::ReduceMean>({power_m, wrap_type<ov::op::v0::Constant>(last_axis)},
                                                          consumers_count(1),
                                                          {{"keep_dims", true}});
    const auto epsilon_m = wrap_type<ov::op::v0::Constant>(op::Predicate(is_supported_epsilon_constant));
    const auto add_eps_m = wrap_type<ov::op::v1::Add>({mean_m, epsilon_m}, consumers_count(1));
    const auto sqrt_m = wrap_type<ov::op::v0::Sqrt>({add_eps_m}, consumers_count(1));
    const auto divide_m = wrap_type<ov::op::v1::Divide>({query_m, sqrt_m}, consumers_count(1));
    const auto scale_m = wrap_type<ov::op::v1::Multiply>({divide_m, query_weight_m}, consumers_count(1));

    const auto normalized_query_m = rms_norm_m | scale_m;

    // QSAPrepare state inputs.
    const auto raw_keys_m = any_input(rank_equals(3));
    const auto cache_m = any_input();
    const auto key_length_m = any_input();
    const auto rope_cos_m = any_input();
    const auto rope_sin_m = any_input();
    const auto key_weight_m = any_input(enabled_norm_weight);
    const auto incomplete_keys_m = any_input();
    const auto incomplete_rope_m = any_input();
    const auto prepare_m = wrap_type_strict_index<QSAPrepare>(
        {raw_keys_m, cache_m, key_length_m, rope_cos_m, rope_sin_m, key_weight_m, incomplete_keys_m, incomplete_rope_m},
        op::Predicate(all_outputs_have_single_consumer) &&
            [](const ov::Output<ov::Node>& output) {
                return is_supported_epsilon(ov::as_type<QSAPrepare>(output.get_node())->get_epsilon());
            },
        {{"key_block_size", QSAIndexer::supported_key_block_size}});

    // `key_length_m` is shared with QSAPrepare, so both ops must consume the same key length.
    const auto indexer_m = wrap_type<SparseAttentionIndexer>(
        {normalized_query_m, prepare_m->output(0), key_length_m},
        [](const ov::Output<ov::Node>& output) {
            const auto indexer = ov::as_type<SparseAttentionIndexer>(output.get_node());
            const auto token_budget = indexer->get_token_budget();
            return token_budget > 0 && token_budget <= QSAIndexer::max_token_budget &&
                   token_budget % indexer->get_key_block_size() == 0;
        },
        {{"key_block_size", QSAIndexer::supported_key_block_size},
         {"num_selection_heads", QSAIndexer::supported_num_selection_heads}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer = ov::as_type_ptr<SparseAttentionIndexer>(pattern_map.at(indexer_m).get_node_shared_ptr());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexer: matched node is expected to be SparseAttentionIndexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        const auto prepare = ov::as_type_ptr<QSAPrepare>(m.get_pattern_map().at(prepare_m));
        OPENVINO_ASSERT(prepare, "FuseQSAIndexer: matched node is expected to be QSAPrepare");

        const auto& query = pattern_map.at(query_m);
        const auto& raw_keys = pattern_map.at(raw_keys_m);
        // Batch and sequence may be dynamic without symbols, which shape_matches cannot relate across inputs;
        // QSAIndexer validation would throw on a mismatch, so skip such graphs instead.
        const auto& query_shape = query.get_partial_shape();
        const auto& keys_shape = raw_keys.get_partial_shape();
        if (!query_shape[0].compatible(keys_shape[0]) || !query_shape[2].compatible(keys_shape[1])) {
            return false;
        }

        const bool is_rms_norm_op = pattern_map.count(rms_norm_m) > 0;
        float query_epsilon = 0.0f;
        if (is_rms_norm_op) {
            const auto rms_norm =
                ov::as_type_ptr<ov::op::internal::RMSNorm>(pattern_map.at(rms_norm_m).get_node_shared_ptr());
            OPENVINO_ASSERT(rms_norm, "FuseQSAIndexer: matched node is expected to be RMSNorm");
            query_epsilon = static_cast<float>(rms_norm->get_epsilon());
        } else {
            const auto epsilon = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(epsilon_m).get_node_shared_ptr());
            OPENVINO_ASSERT(ov::op::util::get_single_value(epsilon, query_epsilon),
                            "FuseQSAIndexer: matched epsilon is expected to be a single-value Constant");
        }

        ov::op::internal::QSAIndexer::Config config;
        config.key_block_size = prepare->get_key_block_size();
        config.token_budget = indexer->get_token_budget();
        config.num_selection_heads = indexer->get_num_selection_heads();
        config.q_epsilon = query_epsilon;
        config.k_epsilon = prepare->get_epsilon();
        config.is_causal = true;

        const auto fused =
            std::make_shared<ov::op::internal::QSAIndexer>(ov::OutputVector{query,
                                                                            raw_keys,
                                                                            pattern_map.at(rope_cos_m),
                                                                            pattern_map.at(rope_sin_m),
                                                                            pattern_map.at(key_length_m),
                                                                            pattern_map.at(query_weight_m),
                                                                            pattern_map.at(key_weight_m),
                                                                            pattern_map.at(cache_m),
                                                                            pattern_map.at(incomplete_keys_m),
                                                                            pattern_map.at(incomplete_rope_m)},
                                                           config);
        fused->set_friendly_name(indexer->get_friendly_name());

        ov::NodeVector fused_nodes{prepare, indexer};
        for (const auto& norm_label : {rms_norm_m, power_m, mean_m, add_eps_m, sqrt_m, divide_m, scale_m}) {
            if (const auto it = pattern_map.find(norm_label); it != pattern_map.end()) {
                fused_nodes.push_back(it->second.get_node_shared_ptr());
            }
        }
        ov::copy_runtime_info(fused_nodes, fused);

        // replace_node/replace_output_update_name do not support replacing two multi-output producers with one node.
        // Output::replace moves consumers, tensor names, and output runtime info.
        indexer->output(0).replace(fused->output(0));
        indexer->output(1).replace(fused->output(1));
        prepare->output(0).replace(fused->output(2));
        prepare->output(1).replace(fused->output(3));
        prepare->output(2).replace(fused->output(4));
        return true;
    };

    register_matcher(std::make_shared<Matcher>(indexer_m, matcher_name), callback);
}
