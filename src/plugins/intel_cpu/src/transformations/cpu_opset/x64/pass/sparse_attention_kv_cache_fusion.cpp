// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_kv_cache_fusion.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/sparse_attention_with_kv_cache.hpp"

namespace ov::intel_cpu {

namespace {

namespace pattern = ov::pass::pattern;

// The nodes of the pattern the rewrite reads from the match.
struct KVCachePattern {
    std::shared_ptr<ov::Node> beam_idx, cur_q, cur_k, cur_v, block_indices, block_count, key_length;
    std::shared_ptr<ov::Node> past_k, past_v, gather_k, gather_v, concat_k, concat_v, sparse;
};

KVCachePattern make_pattern() {
    using namespace ov::pass::pattern;
    KVCachePattern p;
    p.beam_idx = any_input(type_matches(element::i32) && shape_matches("[?]"));
    p.cur_q = any_input();
    p.cur_k = any_input();
    p.cur_v = any_input();
    p.block_indices = any_input();
    p.block_count = any_input();
    p.key_length = any_input();

    // The cache history is only read through ShapeOf, Convert and Gather.
    auto only_cache_consumers = [](const ov::Output<ov::Node>& out) {
        const auto children = out.get_target_inputs();
        return std::all_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
            return ov::is_type_any_of<ov::op::util::ShapeOfBase, ov::op::v0::Convert, ov::op::util::GatherBase>(
                child.get_node());
        });
    };
    p.past_k = wrap_type<ov::op::util::ReadValueBase>(only_cache_consumers);
    p.past_v = wrap_type<ov::op::util::ReadValueBase>(only_cache_consumers);
    // The Convert dies with the fusion, so it must not have other consumers.
    auto convert_past_k = optional<ov::op::v0::Convert>({p.past_k}, consumers_count(1));
    auto convert_past_v = optional<ov::op::v0::Convert>({p.past_v}, consumers_count(1));

    // The beam axis is the batch axis of the cache; both caches are reordered by the same beam_idx.
    auto axis_k = wrap_type<ov::op::v0::Constant>(value_matches("0"));
    auto axis_v = wrap_type<ov::op::v0::Constant>(value_matches("0"));
    const auto beam_gather = attrs_match({{"batch_dims", 0}}) && consumers_count(1);
    p.gather_k = wrap_type<ov::op::v7::Gather, ov::op::v8::Gather>({convert_past_k, p.beam_idx, axis_k}, beam_gather);
    p.gather_v = wrap_type<ov::op::v7::Gather, ov::op::v8::Gather>({convert_past_v, p.beam_idx, axis_v}, beam_gather);

    // key/value are [N, Hkv, S, E] and are concatenated along S. The Concat feeds the SparseAttention, the Assign
    // (maybe through a Convert) and optionally a ShapeOf.
    auto assign_and_optional_shape_of = [](const ov::Output<ov::Node>& out) {
        const auto children = out.get_target_inputs();
        return children.size() == 2 ||
               (children.size() == 3 && std::any_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& c) {
                    return ov::is_type<ov::op::util::ShapeOfBase>(c.get_node());
                }));
    };
    const auto concat_s = rank_equals(4) && attrs_match({{"axis", 2}}) && assign_and_optional_shape_of;
    p.concat_k = wrap_type<ov::op::v0::Concat>({p.gather_k, p.cur_k}, concat_s);
    p.concat_v = wrap_type<ov::op::v0::Concat>({p.gather_v, p.cur_v}, concat_s);

    // The fused op supports the 5/6-input forms only (no scale / sinks) with query_block_size == 1.
    const auto single_query_block = attrs_match({{"query_block_size", 1}});
    auto sparse0 =
        wrap_type<ov::op::v17::SparseAttention>({p.cur_q, p.concat_k, p.concat_v, p.block_indices, p.block_count},
                                                single_query_block);
    auto sparse1 = wrap_type<ov::op::v17::SparseAttention>(
        {p.cur_q, p.concat_k, p.concat_v, p.block_indices, p.block_count, p.key_length},
        single_query_block);
    p.sparse = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sparse0, sparse1});
    return p;
}

// The Assign consuming a Concat directly or through a single Convert; `convert` is that Convert, if any.
struct AssignOfCache {
    ov::op::v6::Assign* assign = nullptr;
    ov::op::v0::Convert* convert = nullptr;
};

AssignOfCache find_assign(const ov::Output<ov::Node>& concat_out) {
    AssignOfCache found;
    for (const auto& target : concat_out.get_target_inputs()) {
        auto* node = target.get_node();
        ov::op::v0::Convert* convert = nullptr;
        if (auto* cvt = ov::as_type<ov::op::v0::Convert>(node)) {
            const auto cvt_targets = cvt->get_output_target_inputs(0);
            if (cvt_targets.size() == 1) {
                node = cvt_targets.begin()->get_node();
                convert = cvt;
            }
        }
        if (auto* assign = ov::as_type<ov::op::v6::Assign>(node)) {
            found.assign = assign;
            found.convert = convert;
            return found;
        }
    }
    return found;
}

// Without moving the ShapeOf consumers of a Concat to the fused op, the dead Concat keeps the old Gather alive as a
// MemoryInput child, and MatchSparseSdpaKvCache rejects children other than {SparseSDPA, ShapeOf}.
void reroute_shape_of_consumers(const std::shared_ptr<ov::Node>& concat, const ov::Output<ov::Node>& replacement) {
    std::vector<ov::Input<ov::Node>> shape_of_inputs;
    for (const auto& target : concat->get_output_target_inputs(0)) {
        if (ov::is_type<ov::op::util::ShapeOfBase>(target.get_node())) {
            shape_of_inputs.push_back(target);
        }
    }
    for (const auto& input : shape_of_inputs) {
        input.replace_source_output(replacement);
    }
}

}  // namespace

bool SparseAttentionKVCacheFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SparseAttentionKVCacheFusion);

    const auto p = make_pattern();
    // Variables whose SparseAttention has already been fused; later matches on the same Variable are reader-only
    // consumers and must be skipped.
    std::unordered_set<std::string> processed_k_variable_ids;
    std::unordered_set<std::string> processed_v_variable_ids;
    bool rewritten = false;

    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v17::SparseAttention>(op)) {
            continue;
        }
        pattern::Matcher matcher(p.sparse, "SparseAttentionKVCacheFusion");
        if (!matcher.match(op->output(0))) {
            continue;
        }
        const auto& pattern_map = matcher.get_pattern_value_map();

        const auto sparse_node = ov::as_type_ptr<ov::op::v17::SparseAttention>(op);
        const auto concat_k_node =
            ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(p.concat_k).get_node_shared_ptr());
        const auto concat_v_node =
            ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(p.concat_v).get_node_shared_ptr());
        const auto gather_k_node =
            ov::as_type_ptr<ov::op::util::GatherBase>(pattern_map.at(p.gather_k).get_node_shared_ptr());
        const auto gather_v_node =
            ov::as_type_ptr<ov::op::util::GatherBase>(pattern_map.at(p.gather_v).get_node_shared_ptr());
        const auto past_k_node =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(p.past_k).get_node_shared_ptr());
        const auto past_v_node =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(p.past_v).get_node_shared_ptr());
        OPENVINO_ASSERT(sparse_node && concat_k_node && concat_v_node && gather_k_node && gather_v_node &&
                            past_k_node && past_v_node,
                        "SparseAttentionKVCacheFusion: the pattern matched nodes of unexpected types");

        if (processed_k_variable_ids.count(past_k_node->get_variable_id()) != 0 ||
            processed_v_variable_ids.count(past_v_node->get_variable_id()) != 0) {
            continue;
        }

        const auto assign_k = find_assign(concat_k_node);
        const auto assign_v = find_assign(concat_v_node);
        if (assign_k.assign == nullptr || past_k_node->get_variable_id() != assign_k.assign->get_variable_id() ||
            assign_v.assign == nullptr || past_v_node->get_variable_id() != assign_v.assign->get_variable_id()) {
            continue;
        }

        OutputVector args = {pattern_map.at(p.cur_q),
                             pattern_map.at(p.cur_k),
                             pattern_map.at(p.cur_v),
                             pattern_map.at(p.block_indices),
                             pattern_map.at(p.block_count)};
        if (pattern_map.count(p.key_length)) {
            args.push_back(pattern_map.at(p.key_length));
        }
        args.push_back(pattern_map.at(p.beam_idx));
        args.push_back(gather_k_node->input_value(0));
        args.push_back(gather_v_node->input_value(0));

        SparseAttentionWithKVCache::Config config;
        config.key_block_size = sparse_node->get_key_block_size();
        config.query_block_size = sparse_node->get_query_block_size();
        config.causal = sparse_node->get_causal();

        auto new_node = std::make_shared<SparseAttentionWithKVCache>(args, config);
        new_node->set_friendly_name(sparse_node->get_friendly_name());
        copy_runtime_info(sparse_node, new_node);
        ov::replace_node(sparse_node, {new_node->output(0)});

        // The new present key / value feed the Assign sinks of the states.
        if (assign_k.convert) {
            assign_k.convert->set_arguments({new_node->output(1)});
        } else {
            assign_k.assign->set_arguments({new_node->output(1)});
        }
        if (assign_v.convert) {
            assign_v.convert->set_arguments({new_node->output(2)});
        } else {
            assign_v.assign->set_arguments({new_node->output(2)});
        }
        reroute_shape_of_consumers(concat_k_node, new_node->output(1));
        reroute_shape_of_consumers(concat_v_node, new_node->output(2));

        // ReadValue -> [Convert] -> SparseAttentionWithKVCache -> [Convert] -> Assign must stay a plain
        // MemoryInput/MemoryOutput pair for MatchSparseSdpaKvCache, so no init subgraph is fused into it.
        past_k_node->get_rt_info()["DisableInitSubgraphFusing"] = true;
        past_v_node->get_rt_info()["DisableInitSubgraphFusing"] = true;

        processed_k_variable_ids.insert(past_k_node->get_variable_id());
        processed_v_variable_ids.insert(past_v_node->get_variable_id());
        rewritten = true;
    }
    return rewritten;
}

}  // namespace ov::intel_cpu
