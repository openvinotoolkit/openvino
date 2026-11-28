// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_kv_cache_fusion.hpp"

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
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
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/common/op/sparse_attention_with_kv_cache.hpp"

using namespace ov::pass;

namespace ov::intel_cpu {

SparseAttentionKVCacheFusion::SparseAttentionKVCacheFusion() {
    MATCHER_SCOPE(SparseAttentionKVCacheFusion);
    using namespace ov::pass::pattern;

    auto beam_idx = any_input(type_matches(element::i32) && shape_matches("[?]"));
    auto cur_q = any_input();
    auto cur_k = any_input();
    auto cur_v = any_input();
    auto block_indices = any_input();
    auto block_count = any_input();
    auto key_length = any_input();

    // The cache history is only read through ShapeOf, Convert and Gather.
    auto only_cache_consumers = [](const ov::Output<ov::Node>& out) {
        const auto children = out.get_target_inputs();
        return std::all_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
            return ov::is_type_any_of<ov::op::util::ShapeOfBase, ov::op::v0::Convert, ov::op::util::GatherBase>(
                child.get_node());
        });
    };
    auto past_k = wrap_type<ov::op::util::ReadValueBase>(only_cache_consumers);
    auto past_v = wrap_type<ov::op::util::ReadValueBase>(only_cache_consumers);
    // The Convert dies with the fusion, so it must not have other consumers.
    auto convert_past_k = optional<ov::op::v0::Convert>({past_k}, consumers_count(1));
    auto convert_past_v = optional<ov::op::v0::Convert>({past_v}, consumers_count(1));

    // The beam axis is the batch axis of the cache; both caches are reordered by the same beam_idx (the label).
    auto beam_gather = consumers_count(1) && [](const ov::Output<ov::Node>& out) {
        const auto* gather = ov::as_type<ov::op::util::GatherBase>(out.get_node());
        return gather != nullptr && gather->get_batch_dims() == 0;
    };
    auto axis_k = wrap_type<ov::op::v0::Constant>(value_matches("0"));
    auto axis_v = wrap_type<ov::op::v0::Constant>(value_matches("0"));
    auto gather_k = wrap_type<ov::op::util::GatherBase>({convert_past_k, beam_idx, axis_k}, beam_gather);
    auto gather_v = wrap_type<ov::op::util::GatherBase>({convert_past_v, beam_idx, axis_v}, beam_gather);

    // key/value are [N, Hkv, S, E] and are concatenated along S (the axis may be negative). The Concat feeds the
    // SparseAttention, the Assign (maybe through a Convert) and optionally a ShapeOf.
    auto concat_s = rank_equals(4) && [](const ov::Output<ov::Node>& out) {
        const auto* concat = ov::as_type<ov::op::v0::Concat>(out.get_node());
        if (concat == nullptr || (concat->get_axis() != 2 && concat->get_axis() != -2)) {
            return false;
        }
        const auto children = out.get_target_inputs();
        return children.size() == 2 ||
               (children.size() == 3 && std::any_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& c) {
                    return ov::is_type<ov::op::util::ShapeOfBase>(c.get_node());
                }));
    };
    auto concat_k = wrap_type<ov::op::v0::Concat>({gather_k, cur_k}, concat_s);
    auto concat_v = wrap_type<ov::op::v0::Concat>({gather_v, cur_v}, concat_s);

    // The fused op supports the 5/6-input forms only (no scale / sinks) with query_block_size == 1.
    auto single_query_block = [](const ov::Output<ov::Node>& out) {
        const auto* sparse = ov::as_type<ov::op::v17::SparseAttention>(out.get_node());
        return sparse != nullptr && sparse->get_query_block_size() == 1;
    };
    auto sparse0 = wrap_type<ov::op::v17::SparseAttention>({cur_q, concat_k, concat_v, block_indices, block_count},
                                                           single_query_block);
    auto sparse1 =
        wrap_type<ov::op::v17::SparseAttention>({cur_q, concat_k, concat_v, block_indices, block_count, key_length},
                                                single_query_block);
    auto sparse = sparse0 | sparse1;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto sparse_node = ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root());
        const auto concat_k_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        const auto gather_k_node =
            ov::as_type_ptr<ov::op::util::GatherBase>(pattern_map.at(gather_k).get_node_shared_ptr());
        const auto gather_v_node =
            ov::as_type_ptr<ov::op::util::GatherBase>(pattern_map.at(gather_v).get_node_shared_ptr());
        const auto past_k_node =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node =
            ov::as_type_ptr<ov::op::util::ReadValueBase>(pattern_map.at(past_v).get_node_shared_ptr());
        if (m_processed_k_variable_ids.count(past_k_node->get_variable_id()) != 0 ||
            m_processed_v_variable_ids.count(past_v_node->get_variable_id()) != 0) {
            return false;
        }

        auto find_assign =
            [&](const ov::Output<ov::Node>& out, ov::op::v6::Assign*& assign, ov::op::v0::Convert*& cvt) {
                for (const auto& to : out.get_target_inputs()) {
                    auto* to_node = to.get_node();
                    if (auto* convert = ov::as_type<ov::op::v0::Convert>(to_node)) {
                        auto cvt_targets = convert->get_output_target_inputs(0);
                        if (cvt_targets.size() == 1) {
                            to_node = cvt_targets.begin()->get_node();
                            cvt = convert;
                        }
                    }
                    assign = ov::as_type<ov::op::v6::Assign>(to_node);
                    if (assign) {
                        return true;
                    }
                }
                return false;
            };
        ov::op::v6::Assign* assign_k_node = nullptr;
        ov::op::v6::Assign* assign_v_node = nullptr;
        ov::op::v0::Convert* assign_cvt_k_node = nullptr;
        ov::op::v0::Convert* assign_cvt_v_node = nullptr;
        if (!find_assign(concat_k_node, assign_k_node, assign_cvt_k_node) ||
            past_k_node->get_variable_id() != assign_k_node->get_variable_id()) {
            return false;
        }
        if (!find_assign(concat_v_node, assign_v_node, assign_cvt_v_node) ||
            past_v_node->get_variable_id() != assign_v_node->get_variable_id()) {
            return false;
        }

        OutputVector args = {pattern_map.at(cur_q),
                             pattern_map.at(cur_k),
                             pattern_map.at(cur_v),
                             pattern_map.at(block_indices),
                             pattern_map.at(block_count)};
        if (pattern_map.count(key_length)) {
            args.push_back(pattern_map.at(key_length));
        }
        args.push_back(pattern_map.at(beam_idx));
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

        if (assign_cvt_k_node) {
            assign_cvt_k_node->set_arguments({new_node->output(1)});
        } else {
            assign_k_node->set_arguments({new_node->output(1)});
        }
        if (assign_cvt_v_node) {
            assign_cvt_v_node->set_arguments({new_node->output(2)});
        } else {
            assign_v_node->set_arguments({new_node->output(2)});
        }

        // Without this reroute the dead Concat keeps the old Gather alive as a MemoryInput child, and
        // MatchSparseSdpaKvCache rejects non-{SparseSDPA, ShapeOf} children.
        auto reroute_shape_of_consumers = [](const std::shared_ptr<Node>& concat_node,
                                             const ov::Output<ov::Node>& replacement) {
            std::vector<ov::Input<Node>> shape_of_inputs;
            for (const auto& target : concat_node->get_output_target_inputs(0)) {
                if (ov::is_type<ov::op::util::ShapeOfBase>(target.get_node())) {
                    shape_of_inputs.push_back(target);
                }
            }
            for (const auto& inp : shape_of_inputs) {
                inp.replace_source_output(replacement);
            }
        };
        reroute_shape_of_consumers(concat_k_node, new_node->output(1));
        reroute_shape_of_consumers(concat_v_node, new_node->output(2));

        // ReadValue -> [Convert] -> SparseAttentionWithKVCache -> [Convert] -> Assign must stay a plain
        // MemoryInput/MemoryOutput pair for MatchSparseSdpaKvCache, so no init subgraph is fused into it.
        past_k_node->get_rt_info()["DisableInitSubgraphFusing"] = true;
        past_v_node->get_rt_info()["DisableInitSubgraphFusing"] = true;

        m_processed_k_variable_ids.insert(past_k_node->get_variable_id());
        m_processed_v_variable_ids.insert(past_v_node->get_variable_id());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sparse, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
