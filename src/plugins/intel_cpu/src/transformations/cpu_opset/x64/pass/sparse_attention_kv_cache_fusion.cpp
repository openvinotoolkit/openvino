// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_kv_cache_fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/sparse_attention_with_kv_cache.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

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

    auto past_k = wrap_type<ov::op::v6::ReadValue>();
    auto past_v = wrap_type<ov::op::v6::ReadValue>();
    auto convert_past_k = wrap_type<ov::op::v0::Convert>({past_k});
    auto convert_past_v = wrap_type<ov::op::v0::Convert>({past_v});

    auto axis_k = wrap_type<ov::op::v0::Constant>();
    auto axis_v = wrap_type<ov::op::v0::Constant>();
    auto gather_k = wrap_type<ov::op::v8::Gather>({past_k | convert_past_k, beam_idx, axis_k}, {{"batch_dims", 0}});
    auto gather_v = wrap_type<ov::op::v8::Gather>({past_v | convert_past_v, beam_idx, axis_v}, {{"batch_dims", 0}});

    auto concat_k = wrap_type<ov::op::v0::Concat>({gather_k, cur_k});
    auto concat_v = wrap_type<ov::op::v0::Concat>({gather_v, cur_v});

    auto sparse0 = wrap_type<ov::op::v17::SparseAttention>({cur_q, concat_k, concat_v, block_indices, block_count});
    auto sparse1 =
        wrap_type<ov::op::v17::SparseAttention>({cur_q, concat_k, concat_v, block_indices, block_count, key_length});
    auto sparse = sparse0 | sparse1;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto sparse_node = ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root());
        // the fused op supports the 5/6-input forms only (no scale / sinks) with query_block_size == 1
        if (!sparse_node || sparse_node->get_input_size() > 6 || sparse_node->get_query_block_size() != 1) {
            return false;
        }

        const auto concat_k_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        if (concat_k_node->get_axis() != 2 || concat_v_node->get_axis() != 2) {
            return false;
        }
        for (const auto& concat : {concat_k_node, concat_v_node}) {
            const auto& ps = concat->get_output_partial_shape(0);
            if (ps.rank().is_dynamic() || ps.size() != 4) {
                return false;
            }
        }

        // the beam axis is the batch axis of the cache
        for (const auto& axis_label : {axis_k, axis_v}) {
            const auto axis_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(axis_label).get_node_shared_ptr());
            if (!axis_const || ov::shape_size(axis_const->get_shape()) != 1 ||
                axis_const->cast_vector<int64_t>()[0] != 0) {
                return false;
            }
        }

        const auto past_k_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        if (m_processed_k_variable_ids.count(past_k_node->get_variable_id()) ||
            m_processed_v_variable_ids.count(past_v_node->get_variable_id())) {
            return false;
        }

        auto check_valid_children_type = [](const ov::Output<ov::Node>& out) {
            auto children = out.get_target_inputs();
            return std::all_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
                auto* node = child.get_node();
                return any_of(node->get_type_info(),
                              ov::op::v0::ShapeOf::get_type_info_static(),
                              ov::op::v3::ShapeOf::get_type_info_static(),
                              ov::op::v0::Convert::get_type_info_static(),
                              ov::op::v8::Gather::get_type_info_static());
            });
        };
        if (!check_valid_children_type(past_k_node) || !check_valid_children_type(past_v_node)) {
            return false;
        }

        // Concat: the SparseAttention, the Assign and optionally a ShapeOf
        for (const auto& item : {concat_k_node, concat_v_node}) {
            auto&& children = item->get_output_target_inputs(0);
            if (children.size() == 3) {
                if (!std::any_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
                        return ov::is_type_any_of<ov::op::v3::ShapeOf, ov::op::v0::ShapeOf>(child.get_node());
                    })) {
                    return false;
                }
            } else if (children.size() != 2) {
                return false;
            }
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

        // the dead-after-fusion nodes must not have other consumers
        auto is_optional_one_child = [&pattern_map](const std::shared_ptr<Node>& node) {
            if (pattern_map.count(node)) {
                return pattern_map.at(node).get_node_shared_ptr()->get_output_target_inputs(0).size() == 1;
            }
            return true;
        };
        if (!is_optional_one_child(convert_past_k) || !is_optional_one_child(convert_past_v)) {
            return false;
        }

        // past_k & past_v must be reordered by the same beam_idx
        const auto gather_k_node = ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_k).get_node_shared_ptr());
        const auto gather_v_node = ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_v).get_node_shared_ptr());
        if (gather_k_node->input_value(1) != gather_v_node->input_value(1) ||
            gather_k_node->get_output_target_inputs(0).size() != 1 ||
            gather_v_node->get_output_target_inputs(0).size() != 1) {
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
