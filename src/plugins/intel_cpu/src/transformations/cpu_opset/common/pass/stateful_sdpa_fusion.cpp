// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_sdpa_fusion.hpp"

#include <utils/general_utils.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/defs.hpp"
#include "transformations/symbolic_transformations/symbol_optimization.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/transpose_sinking/ts_shape_of.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "transformations/cpu_opset/x64/pass/sdpa_fuse_transpose_reshape.hpp"
#endif

using namespace ov::pass;

namespace ov::intel_cpu {

StatefulSDPAFusion::StatefulSDPAFusion() {
    MATCHER_SCOPE(StatefulSDPAFusion);
    using namespace ov::pass::pattern;

    auto beam_idx = any_input(type_matches(element::i32) && shape_matches("[?]"));
    auto cur_q = any_input();
    auto cur_k = any_input();
    auto cur_v = any_input();

    auto past_k = wrap_type<ov::op::v6::ReadValue>();
    auto past_v = wrap_type<ov::op::v6::ReadValue>();

    auto convert_past_k = wrap_type<ov::op::v0::Convert>({past_k});
    auto convert_past_v = wrap_type<ov::op::v0::Convert>({past_v});

    auto gather_input_k =
        wrap_type<ov::op::v8::Gather>({past_k | convert_past_k, beam_idx, "axis_beam"}, {{"batch_dims", 0}});
    auto gather_input_v =
        wrap_type<ov::op::v8::Gather>({past_v | convert_past_v, beam_idx, "axis_beam"}, {{"batch_dims", 0}});

    auto concat_k = wrap_type<ov::op::v0::Concat>({gather_input_k, cur_k});
    auto concat_v = wrap_type<ov::op::v0::Concat>({gather_input_v, cur_v});

    std::shared_ptr<Node> reshape_k;
    std::shared_ptr<Node> reshape_v;
    std::shared_ptr<Node> unsqueeze_k;
    std::shared_ptr<Node> unsqueeze_v;
    std::shared_ptr<Node> computed_bcst_k;
    std::shared_ptr<Node> computed_bcst_v;
    std::shared_ptr<Node> multiply_k;
    std::shared_ptr<Node> multiply_v;
    std::shared_ptr<Node> mq_reshape_k;
    std::shared_ptr<Node> mq_reshape_v;
    std::shared_ptr<Node> computed_bcst3_k;
    std::shared_ptr<Node> computed_bcst3_v;
    auto multi_query_bcst = [](const std::shared_ptr<Node>& kv) {
        auto reshape_kv = wrap_type<ov::op::v1::Reshape>({kv, any_input()});
        auto unsqueeze_kv = wrap_type<ov::op::v0::Unsqueeze>({kv, any_input()});

        auto check_one = [](const Output<Node>& output) -> bool {
            auto node = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
            if (!node) {
                return false;
            }
            const auto& bcst_arg = node->cast_vector<float>();
            return std::all_of(bcst_arg.begin(), bcst_arg.end(), [](float i) {
                return i == 1.0F;
            });
        };
        auto constant_bcst = wrap_type<ov::op::v0::Constant>(check_one);

        auto computed_bcst =
            wrap_type<ov::op::v1::Broadcast>({constant_bcst, any_input(), any_input()}, {{"mode", "numpy"}});

        auto multiply_kv = wrap_type<ov::op::v1::Multiply>({reshape_kv | unsqueeze_kv, constant_bcst | computed_bcst});
        auto computed_bcst3 =
            wrap_type<ov::op::v3::Broadcast>({unsqueeze_kv, any_input()}, {{"mode", "bidirectional"}});

        auto result = wrap_type<ov::op::v1::Reshape>({multiply_kv | computed_bcst3, any_input()});
        return std::make_tuple(result, reshape_kv, unsqueeze_kv, computed_bcst, multiply_kv, computed_bcst3);
    };

    std::tie(mq_reshape_k, reshape_k, unsqueeze_k, computed_bcst_k, multiply_k, computed_bcst3_k) =
        multi_query_bcst(concat_k);
    std::tie(mq_reshape_v, reshape_v, unsqueeze_v, computed_bcst_v, multiply_v, computed_bcst3_v) =
        multi_query_bcst(concat_v);
    auto present_k = concat_k | mq_reshape_k;
    auto present_v = concat_v | mq_reshape_v;

    // canonical q/k/v shape definition: [B,H,...L,S]
    auto sdp0 = wrap_type<ov::op::v13::ScaledDotProductAttention>({cur_q, present_k, present_v});
    auto sdp1 = wrap_type<ov::op::v13::ScaledDotProductAttention>({cur_q, present_k, present_v, any_input()});
    auto sdp2 =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({cur_q, present_k, present_v, any_input(), any_input()});

    // non-canonical q/k/v shape definitions, for example: [L, B, H, S]/[B, L, H, S]
    auto order_k = wrap_type<ov::op::v0::Constant>();
    auto order_v = wrap_type<ov::op::v0::Constant>();
    auto order_q = wrap_type<ov::op::v0::Constant>();
    auto transpose_q = wrap_type<ov::op::v1::Transpose>({cur_q, order_q});
    auto transpose_k = wrap_type<ov::op::v1::Transpose>({present_k, order_k});
    auto transpose_v = wrap_type<ov::op::v1::Transpose>({present_v, order_v});

    auto sdp_trans0 = wrap_type<ov::op::v13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v});
    auto sdp_trans1 =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input()});
    auto sdp_trans2 = wrap_type<ov::op::v13::ScaledDotProductAttention>(
        {transpose_q, transpose_k, transpose_v, any_input(), any_input()});

    auto sdp = sdp0 | sdp1 | sdp2 | sdp_trans0 | sdp_trans1 | sdp_trans2;

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        // Check concat axes equality first
        const auto concat_k_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        if (concat_k_node->get_axis() != concat_v_node->get_axis()) {
            return false;
        }

        auto find_assign =
            [&](const ov::Output<ov::Node>& out, ov::op::v6::Assign*& assign, ov::op::v0::Convert*& cvt) {
                auto present_to = out.get_target_inputs();
                for (const auto& to : present_to) {
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
        auto check_valid_children_type = [](const ov::Output<ov::Node>& out) {
            auto children = out.get_target_inputs();
            return std::all_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
                auto* node = child.get_node();
                return any_of(node->get_type_info(),
                              ov::op::v13::ScaledDotProductAttention::get_type_info_static(),
                              ov::op::v0::ShapeOf::get_type_info_static(),
                              ov::op::v3::ShapeOf::get_type_info_static(),
                              ov::op::v0::Convert::get_type_info_static(),
                              ov::op::v8::Gather::get_type_info_static());
            });
        };

        const auto sdp_node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        if (!check_valid_children_type(past_k_node) || !check_valid_children_type(past_v_node)) {
            return false;
        }
        for (auto&& item : {concat_k_node, concat_v_node}) {
            auto&& children = item->get_output_target_inputs(0);
            switch (children.size()) {
            case 2:
                // pass, as the existence of Assign will be checked later
                break;
            case 3:
                // the first one leads to SDPA, otherwise the matcher doesn't find the pattern
                // the second one leads to Assign, and this is checked later
                // the third child is allowed to be a ShapeOf op only, thus one of them must be ShapeOf
                if (!std::any_of(children.begin(), children.end(), [](const ov::Input<ov::Node>& child) {
                        return ov::is_type_any_of<ov::op::v3::ShapeOf, ov::op::v0::ShapeOf>(child.get_node());
                    })) {
                    return false;
                }
                break;
            default:
                return false;
            }
        }

        ov::op::v6::Assign* assign_k_node = nullptr;
        ov::op::v6::Assign* assign_v_node = nullptr;
        ov::op::v0::Convert* assign_cvt_k_node = nullptr;
        ov::op::v0::Convert* assign_cvt_v_node = nullptr;
        if (!find_assign(concat_k_node, assign_k_node, assign_cvt_k_node)) {
            return false;
        }
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id()) {
            return false;
        }

        if (!find_assign(concat_v_node, assign_v_node, assign_cvt_v_node)) {
            return false;
        }
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id()) {
            return false;
        }

        auto is_optional_one_child = [&pattern_map](const std::vector<std::shared_ptr<Node>>& nodes) {
            return std::all_of(nodes.begin(), nodes.end(), [&](const std::shared_ptr<Node>& node) {
                if (pattern_map.count(node)) {
                    auto p = pattern_map.at(node).get_node_shared_ptr();
                    return p->get_output_target_inputs(0).size() == 1;
                }
                return true;
            });
        };
        if (!is_optional_one_child({convert_past_k,
                                    convert_past_v,
                                    transpose_q,
                                    transpose_k,
                                    transpose_v,
                                    reshape_k,
                                    unsqueeze_k,
                                    computed_bcst_k,
                                    multiply_k,
                                    reshape_v,
                                    unsqueeze_v,
                                    computed_bcst_v,
                                    multiply_v,
                                    mq_reshape_k,
                                    mq_reshape_v,
                                    computed_bcst3_k,
                                    computed_bcst3_v})) {
            return false;
        }

        // past_k & past_v must be reordered by same beam_idx
        const auto gather_k_node =
            ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_input_k).get_node_shared_ptr());
        const auto gather_v_node =
            ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_input_v).get_node_shared_ptr());
        if (gather_k_node->input_value(1) != gather_v_node->input_value(1) ||
            gather_k_node->get_output_target_inputs(0).size() != 1 ||
            gather_v_node->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        OutputVector args = sdp_node->input_values();
        args[0] = pattern_map.at(cur_q);
        args[1] = pattern_map.at(cur_k);
        args[2] = pattern_map.at(cur_v);
        args.push_back(pattern_map.at(beam_idx));
        args.push_back(gather_k_node->input_value(0));
        args.push_back(gather_v_node->input_value(0));
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;

        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;

        if (pattern_map.count(order_q) && pattern_map.count(order_k) && pattern_map.count(order_v)) {
            const auto order_q_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(order_q).get_node_shared_ptr());
            const auto order_k_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(order_k).get_node_shared_ptr());
            const auto order_v_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(order_v).get_node_shared_ptr());
            const auto& permute_q = order_q_node->cast_vector<int32_t>();
            const auto& permute_k = order_k_node->cast_vector<int32_t>();
            const auto& permute_v = order_v_node->cast_vector<int32_t>();
            if (permute_q != permute_k || permute_q != permute_v) {
                return false;
            }
            config.permute_axes.resize(permute_q.size());
            for (size_t i = 0; i < permute_q.size(); i++) {
                config.permute_axes[i] = static_cast<size_t>(permute_q[i]);
            }
        }

        const auto& old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        copy_runtime_info(old_node, new_node);
        ov::replace_node(old_node, {new_node->output(0)});
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

        // Markup pattern:
        // ReadValue->Convert(Optional)->ScaledDotProductAttentionWithKVCache->Convert(Optional)->Assign, so that
        // ReadValue can't be replaced with ReadValueWithSubgraph in this pattern.
        // TODO: Temporarily skip this pattern. If MemoryInputSDPA supports Subgraph in the future, it may be deleted.
        past_k_node->get_rt_info()["DisableInitSubgraphFusing"] = true;
        past_v_node->get_rt_info()["DisableInitSubgraphFusing"] = true;

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp, matcher_name);
    this->register_matcher(m, callback);
}

bool SDPASubgraphFusion::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(SDPASubgraphFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false);
    auto& ctx_manager = *symbolic_optimizations.get_manager();

    CPU_REGISTER_PASS_COMMON(ctx_manager, ov::pass::SimplifyGatherShapeOf);
    CPU_REGISTER_PASS_COMMON(ctx_manager, ov::pass::transpose_sinking::TSShapeOfForward);
    ctx_manager.register_pass<StatefulSDPAFusion>();
    CPU_REGISTER_PASS_X64(ctx_manager, ov::intel_cpu::SDPAFuseTransposeReshape);

    return symbolic_optimizations.run_on_model(f);
}

}  // namespace ov::intel_cpu
