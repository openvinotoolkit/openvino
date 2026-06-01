// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

SDPA::SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto past_k_in = opp::wrap_type<ov::op::v0::Parameter>();
    // Optional beam-search gather: Parameter → Gather(beam_idx) → ... (stateless LLM models)
    auto past_k_gather = opp::optional<ov::op::v8::Gather>({past_k_in->output(0), opp::any_input(), opp::any_input()});
    auto past_k_cvt = opp::optional<ov::op::v0::Convert>({past_k_gather->output(0)});
    auto past_k_cat = opp::wrap_type<ov::op::v0::Concat>({past_k_cvt, opp::any_input()});

    auto past_v_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_v_gather = opp::optional<ov::op::v8::Gather>({past_v_in->output(0), opp::any_input(), opp::any_input()});
    auto past_v_cvt = opp::optional<ov::op::v0::Convert>({past_v_gather->output(0)});
    auto past_v_cat = opp::wrap_type<ov::op::v0::Concat>({past_v_cvt, opp::any_input()});

    // Optional part, probably one of many. Replace by graph traversal!
    auto opt_unsq_k = opp::optional<ov::op::v0::Unsqueeze>({past_k_cat->output(0), opp::any_input()});
    auto opt_bcast_k = opp::optional<ov::op::v3::Broadcast>({opt_unsq_k->output(0), opp::any_input()});
    auto opt_rshp_k = opp::optional<ov::op::v1::Reshape>({opt_bcast_k->output(0), opp::any_input()});

    auto opt_unsq_v = opp::optional<ov::op::v0::Unsqueeze>({past_v_cat->output(0), opp::any_input()});
    auto opt_bcast_v = opp::optional<ov::op::v3::Broadcast>({opt_unsq_v->output(0), opp::any_input()});
    auto opt_rshp_v = opp::optional<ov::op::v1::Reshape>({opt_bcast_v->output(0), opp::any_input()});

    auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>(
        {opp::any_input(), opt_rshp_k, opt_rshp_v, opp::any_input(), opp::any_input()});
    auto trans = opp::wrap_type<ov::op::v1::Transpose>({sdpa, opp::any_input()});
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({trans, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto pattern_nodes = std::vector<std::shared_ptr<ov::Node>>{past_k_in,
                                                                    past_k_gather,
                                                                    past_k_cvt,
                                                                    past_k_cat,
                                                                    past_v_in,
                                                                    past_v_gather,
                                                                    past_v_cvt,
                                                                    past_v_cat,
                                                                    opt_unsq_k,
                                                                    opt_bcast_k,
                                                                    opt_rshp_k,
                                                                    opt_unsq_v,
                                                                    opt_bcast_v,
                                                                    opt_rshp_v,
                                                                    sdpa,
                                                                    trans,
                                                                    reshape};
        for (auto&& pattern_node : pattern_nodes) {
            if (auto match_iter = node_to_output.find(pattern_node); match_iter != node_to_output.end()) {
                auto matched_node = match_iter->second.get_node_shared_ptr();
                if (auto group_iter = node_to_gptr->find(matched_node); group_iter != node_to_gptr->end()) {
                    group_iter->second->isolate(isol_tag);
                }
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(reshape, "TagSDPA"), std::move(callback));
}

/*
    Decomposed SDPA Pattern:
            Convert
                \       /
                 Concat
                    |
                opt:Unsqueeze
                    |
                opt:Broadcast   Convert
                    |       \       /
                opt:Reshape       Concat
        \           /           |
            MatMul           opt:Unsqueeze
    \       /                   |
       Add                   opt:Broadcast
        |                       |
     Softmax                opt:Reshape
            \               /
                  MatMul
                    |
                Transpose
                    |
                Reshape
                    |
*/

SDPADecomposed::SDPADecomposed(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                               const std::string& isol_tag) {
    auto convert1 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto concat1 = opp::wrap_type<ov::op::v0::Concat>({convert1, opp::any_input()});

    // GQA optional nodes — require single consumer so shared KV (e.g. Gemma4) does not
    // accidentally match: if any expansion node is shared across multiple heads the
    // predicate fails and the optional is treated as absent, causing the overall pattern
    // to fall through rather than matching an incorrect multi-branch subgraph.
    auto single_user = [](const ov::Output<ov::Node>& output) {
        return output.get_target_inputs().size() == 1;
    };
    auto unsqueeze1 = opp::optional<ov::op::v0::Unsqueeze>({concat1, opp::any_input()}, single_user);
    auto broadcast1 = opp::optional<ov::op::v3::Broadcast>({unsqueeze1, opp::any_input()}, single_user);
    auto reshape1 = opp::optional<ov::op::v1::Reshape>({broadcast1, opp::any_input()}, single_user);

    auto convert2 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto concat2 = opp::wrap_type<ov::op::v0::Concat>({convert2, opp::any_input()});

    // GQA optional nodes — same single-consumer guard
    auto unsqueeze2 = opp::optional<ov::op::v0::Unsqueeze>({concat2, opp::any_input()}, single_user);
    auto broadcast2 = opp::optional<ov::op::v3::Broadcast>({unsqueeze2, opp::any_input()}, single_user);
    auto reshape2 = opp::optional<ov::op::v1::Reshape>({broadcast2, opp::any_input()}, single_user);

    auto matmul1 = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), reshape1});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul1, opp::any_input()});
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({add});

    auto matmul2 = opp::wrap_type<ov::op::v0::MatMul>({softmax, reshape2});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul2, opp::any_input()});
    auto reshape3 = opp::wrap_type<ov::op::v1::Reshape>({transpose, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        LOG_DEBUG("Decomposed SDPA pattern matched!");

        auto& node_to_output = m.get_pattern_value_map();

        // Helper lambda to extract and isolate matched nodes
        auto isolate_matched = [&](const auto& pattern) {
            auto optional_node = node_to_output.find(pattern);
            if (optional_node != node_to_output.end()) {
                auto matched_node = optional_node->second.get_node_shared_ptr();
                node_to_gptr->at(matched_node)->isolate(isol_tag);
            }
        };

        // Isolate all matched nodes in the pattern
        isolate_matched(convert1);
        isolate_matched(concat1);
        isolate_matched(unsqueeze1);
        isolate_matched(broadcast1);
        isolate_matched(reshape1);

        isolate_matched(convert2);
        isolate_matched(concat2);
        isolate_matched(unsqueeze2);
        isolate_matched(broadcast2);
        isolate_matched(reshape2);

        isolate_matched(matmul1);
        isolate_matched(add);
        isolate_matched(softmax);
        isolate_matched(matmul2);
        isolate_matched(transpose);
        isolate_matched(reshape3);

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape3, "TagSDPADecomposed"), std::move(callback));
}

SDPADecomposed1::SDPADecomposed1(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                 const std::string& isol_tag) {
    auto concat1 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input()});
    auto convert1 = opp::wrap_type<ov::op::v0::Convert>({concat1});
    auto multiply1 = opp::wrap_type<ov::op::v1::Multiply>({convert1, opp::any_input()});
    auto transpose1 = opp::wrap_type<ov::op::v1::Transpose>({multiply1, opp::any_input()});
    auto matmul1 = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), transpose1});

    // auto shape_of = ... (removed: AttentionBroadcast4 pass folds these nodes before pattern matching)
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul1, opp::any_input()});

    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({add});

    auto concat2 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input()});
    auto convert2 = opp::wrap_type<ov::op::v0::Convert>({concat2});
    auto multiply2 = opp::wrap_type<ov::op::v1::Multiply>({convert2, opp::any_input()});

    auto matmul2 = opp::wrap_type<ov::op::v0::MatMul>({softmax, multiply2});
    auto reshape1 = opp::wrap_type<ov::op::v1::Reshape>({matmul2, opp::any_input()});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({reshape1, opp::any_input()});
    auto reshape2 = opp::wrap_type<ov::op::v1::Reshape>({transpose, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    LOG_DEBUG("searching for Decomposed1 SDPA pattern");
    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        LOG_DEBUG("Decomposed1 SDPA pattern matched!");

        auto& node_to_output = m.get_pattern_value_map();

        // Helper lambda to extract and isolate matched nodes
        auto isolate_matched = [&](const auto& pattern) {
            auto optional_node = node_to_output.find(pattern);
            if (optional_node != node_to_output.end()) {
                auto matched_node = optional_node->second.get_node_shared_ptr();
                node_to_gptr->at(matched_node)->isolate(isol_tag);
            }
        };

        node_to_output.at(concat2).get_node()->input(1).get_source_output().get_node()->set_friendly_name("past_key_values.0.value");
        node_to_output.at(concat1).get_node()->input(1).get_source_output().get_node()->set_friendly_name("past_key_values.0.key");

        isolate_matched(concat1);
        isolate_matched(convert1);
        isolate_matched(multiply1);
        isolate_matched(transpose1);
        isolate_matched(matmul1);

        isolate_matched(add);

        isolate_matched(softmax);

        isolate_matched(concat2);
        isolate_matched(convert2);
        isolate_matched(multiply2);

        isolate_matched(matmul2);
        isolate_matched(reshape1);
        isolate_matched(transpose);
        isolate_matched(reshape2);

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape2, "TagSDPADecomposed1"), std::move(callback));
}

}  // namespace attn

namespace regularize {

namespace opp = ov::pass::pattern;

AttentionBroadcast::AttentionBroadcast() {
    // NB(dm): We've seen cases where this dynamic subgraph is placed on the K-path,
    // but I'd expect it could be on the V-path as well - so _kv in the name
    auto past_kv_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_kv_cvt = opp::optional<ov::op::v0::Convert>({past_kv_in->output(0)});
    auto past_kv_cat = opp::wrap_type<ov::op::v0::Concat>({past_kv_cvt, opp::any_input()});

    // The dynamic shape calculation to be eliminated
    // NB: It only works in static shape graphs
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({past_kv_cat});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    // NB: THREE inputs is also a case (see below)
    auto concat = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input(), opp::any_input()});

    // Broadcast - the consumer
    auto unsq_kv = opp::wrap_type<ov::op::v0::Unsqueeze>({past_kv_cat, opp::any_input()});
    auto bcast_kv = opp::wrap_type<ov::op::v3::Broadcast>({unsq_kv, concat});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_gather_out = node_to_output.at(gather);
        if (matched_gather_out.get_target_inputs().size() > 1) {
            // This pattern is for the Gather that feeds a single Concat.
            return false;
        }
        auto matched_concat_out = node_to_output.at(concat);
        auto& matched_concat_tensor = matched_concat_out.get_tensor();
        if (matched_concat_tensor.has_and_set_bound()) {
            // Replace the dynamic shape calculation with a static constant
            // This is bad but it in the current realm it is what it is
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_concat_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_concat_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_concat_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(bcast_kv, "AttentionBroadcast"), std::move(callback));
}

// FIXME: Same as above but Concat has three inputs instead of four
AttentionBroadcast2::AttentionBroadcast2() {
    auto past_kv_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_kv_cvt = opp::optional<ov::op::v0::Convert>({past_kv_in->output(0)});
    auto past_kv_cat = opp::wrap_type<ov::op::v0::Concat>({past_kv_cvt, opp::any_input()});

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({past_kv_cat});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat =
        opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});  // THIS IS the difference

    // FIXME: using past_kv_cat as a 0th argument to this Unsqueeze breaks the pattern
    // for Phi-4
    auto unsq_kv = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::any_input()});
    auto bcast_kv = opp::wrap_type<ov::op::v3::Broadcast>({unsq_kv, concat});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_concat_out = node_to_output.at(concat);
        auto& matched_concat_tensor = matched_concat_out.get_tensor();
        if (matched_concat_tensor.has_and_set_bound()) {
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_concat_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_concat_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_concat_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(bcast_kv, "AttentionBroadcast2"), std::move(callback));
}

// FIXME: Same as AttentionBroadcast but Gather connects to multiple Concats
AttentionBroadcast3::AttentionBroadcast3() {
    // NB(dm): We've seen cases where this dynamic subgraph is placed on the K-path,
    // but I'd expect it could be on the V-path as well - so _kv in the name
    auto past_kv_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_kv_cvt = opp::optional<ov::op::v0::Convert>({past_kv_in->output(0)});
    auto past_kv_cat = opp::wrap_type<ov::op::v0::Concat>({past_kv_cvt, opp::any_input()});

    // The dynamic shape calculation to be eliminated
    // NB: It only works in static shape graphs
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({past_kv_cat});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input(), opp::any_input()});

    // Broadcast - the consumer
    auto unsq_kv = opp::wrap_type<ov::op::v0::Unsqueeze>({past_kv_cat, opp::any_input()});
    auto bcast_kv = opp::wrap_type<ov::op::v3::Broadcast>({unsq_kv, concat});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_gather_out = node_to_output.at(gather);
        if (matched_gather_out.get_target_inputs().size() == 1) {
            // This pattern only for the Gather feeding multiple Concats.
            return false;
        }
        auto& matched_gather_tensor = matched_gather_out.get_tensor();
        if (matched_gather_tensor.has_and_set_bound()) {
            // Replace the dynamic shape calculation with a static constant
            // This is bad but it in the current realm it is what it is
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_gather_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_gather_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_gather_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
            return true;  // root changed
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(bcast_kv, "AttentionBroadcast3"), std::move(callback));
}

ShapeOfParameter::ShapeOfParameter() {
    auto param_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto param_cvt = opp::wrap_type<ov::op::v0::Convert>({param_in});
    auto param_shp = opp::wrap_type<ov::op::v3::ShapeOf>({param_cvt});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        // Replace the path with a known constant too
        auto matched_shape_out = node_to_output.at(param_shp);
        auto& matched_shape_tensor = matched_shape_out.get_tensor();
        if (matched_shape_tensor.has_and_set_bound()) {
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_shape_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_shape_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_shape_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed (?)
    };
    register_matcher(std::make_shared<opp::Matcher>(param_shp, "ShapeOfParameter"), std::move(callback));
}


// AttentionBroadcast4: folds the ShapeOf->Gather->Concat->Reshape chain that
// appears on the attention mask path (originally part of SDPADecomposed1 before
// it was simplified). Running this before SDPADecomposed1 allows that pattern
// to use any_input() for the Add's second operand.
AttentionBroadcast4::AttentionBroadcast4() {
    // Pattern derived from the original SDPADecomposed1 mask-shape sub-graph:
    //   ShapeOf(kv) -> Gather -> Const
    //                         \
    //                          Concat(any, Const, any, Gather) -> Reshape -> Add
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>(opp::any_input());
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto constant = opp::wrap_type<ov::op::v0::Constant>();
    auto concat_gather = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), constant, opp::any_input(), gather});
    auto reshape_gather = opp::wrap_type<ov::op::v1::Reshape>({concat_gather, opp::any_input()});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto shape_of_node = std::dynamic_pointer_cast<ov::op::v3::ShapeOf>(
            node_to_output.at(shape_of).get_node_shared_ptr());
        if (!shape_of_node) {
            return false;
        }
        const auto& partial_shape = shape_of_node->input_value(0).get_partial_shape();
        if (!partial_shape.is_static()) {
            return false;
        }
        const auto shape = partial_shape.to_shape();
        const auto element_type = shape_of_node->output(0).get_element_type();
        auto shape_const = std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape{shape.size()}, shape);
        shape_of_node->output(0).replace(shape_const->output(0));
        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape_gather, "AttentionBroadcast4"),
                     std::move(callback));
}

// SeparateVCache: when a V-cache chain (Concat->Convert->Multiply) is shared
// across multiple MatMul consumers, duplicate it so each consumer owns an
// independent chain. This is required for correct partition-weight bank
// assignment by the NPUW partitioner.
SeparateVCache::SeparateVCache() {
    auto concat1 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input()});
    auto convert1 = opp::wrap_type<ov::op::v0::Convert>({concat1});
    auto multiply1 = opp::wrap_type<ov::op::v1::Multiply>({convert1, opp::any_input()});
    auto transpose1 = opp::wrap_type<ov::op::v1::Transpose>({multiply1, opp::any_input()});
    auto matmul1 = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), transpose1});

    auto add = opp::wrap_type<ov::op::v1::Add>({matmul1, opp::any_input()});
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({add});

    auto concat2 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input()});
    auto convert2 = opp::wrap_type<ov::op::v0::Convert>({concat2});
    auto multiply2 = opp::wrap_type<ov::op::v1::Multiply>({convert2, opp::any_input()});

    auto matmul2 = opp::wrap_type<ov::op::v0::MatMul>({softmax, multiply2});
    auto reshape1 = opp::wrap_type<ov::op::v1::Reshape>({matmul2, opp::any_input()});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({reshape1, opp::any_input()});
    auto reshape2 = opp::wrap_type<ov::op::v1::Reshape>({transpose, opp::any_input()});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto multiply2_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
            node_to_output.at(multiply2).get_node_shared_ptr());
        auto matmul2_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
            node_to_output.at(matmul2).get_node_shared_ptr());

        auto ml_node_target_inputs = multiply2_node->output(0).get_target_inputs();
        if (ml_node_target_inputs.size() <= 1) {
            // multiply2 has only one consumer (matmul2 in this pattern) -- nothing to unshare.
            return false;
        }

        auto concat2_node = std::dynamic_pointer_cast<ov::op::v0::Concat>(
            node_to_output.at(concat2).get_node_shared_ptr());
        auto convert2_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(
            node_to_output.at(convert2).get_node_shared_ptr());

        auto concat2_inputs = concat2_node->inputs();
        auto multiply2_inputs = multiply2_node->inputs();

        // For each extra consumer of multiply2 (beyond the matmul2 already in the pattern),
        // create a duplicate concat->convert->multiply chain and redirect that consumer's
        // V input to the new chain. Each consumer keeps its own Q (its own softmax) --
        // only the shared V-cache side is duplicated.
        for (auto& target_input : ml_node_target_inputs) {
            // Skip the matmul2 that is already part of this matched pattern.
            if (target_input.get_node() == matmul2_node.get()) {
                continue;
            }

            if (!dynamic_cast<ov::op::v0::MatMul*>(target_input.get_node())) {
                continue;
            }

            // Clone concat2 with same inputs
            auto new_concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{concat2_inputs[0].get_source_output(),
                                 concat2_inputs[1].get_source_output()},
                concat2_node->get_axis());

            // Clone convert2
            auto new_convert = std::make_shared<ov::op::v0::Convert>(
                new_concat, convert2_node->get_convert_element_type());

            // Clone multiply2.
            // IMPORTANT: clone the scale constant (not reuse it) so each new chain
            // owns its own constant node. Sharing the same constant across multiple
            // call-site subgraph models confuses the partitioner's propagateWeights
            // bank-assignment, which expects exactly one constant per call site.
            auto scale_source = multiply2_inputs[1].get_source_output();
            auto scale_const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                scale_source.get_node_shared_ptr());
            ov::Output<ov::Node> new_scale_output;
            if (scale_const_node) {
                auto new_scale = std::make_shared<ov::op::v0::Constant>(*scale_const_node);
                new_scale_output = new_scale->output(0);
            } else {
                // Not a plain constant (e.g. a model parameter) -- safe to reuse.
                new_scale_output = scale_source;
            }
            auto new_multiply = std::make_shared<ov::op::v1::Multiply>(new_convert, new_scale_output);

            // Redirect this consumer's V input from the shared multiply2 to the new chain.
            target_input.replace_source_output(new_multiply->output(0));
        }

        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape2, "SeparateVCache"),
                     std::move(callback));
}

bool RegularizeSDPA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool model_changed = false;
    if (m_run_broadcast_pattern) {
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast2>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast3>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast4>();
        rewr.add_matcher<ov::npuw::patterns::regularize::SeparateVCache>();

        model_changed |= rewr.run_on_model(model);
    }

    // FIXME: generally all these patterns are supposed to improve the partitioning - thus
    // the performance. However, ShapeOfParameter seems to be working fine for all known case,
    // while AttentionBroadcast patterns might break the partitioning (related to F16IC).
    ov::pass::GraphRewrite rewr2;
    rewr2.add_matcher<ov::npuw::patterns::regularize::ShapeOfParameter>();
    model_changed |= rewr2.run_on_model(model);

    return model_changed;
}

}  // namespace regularize

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
