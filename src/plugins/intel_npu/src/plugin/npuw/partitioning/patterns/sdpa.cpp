// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include <regex>

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

namespace {
namespace opp = ov::pass::pattern;

// Predicate for opp::wrap_type: returns true when the Add node's second input
// comes from a global attention mask path: Reshape(Tile(Convert(Parameter("..attention_mask_global..")))
bool consumes_global_mask(const ov::Output<ov::Node>& output) {
    auto node_ptr = output.get_node_shared_ptr();
    if (!node_ptr) {
        return false;
    }
    auto reshape = node_ptr->get_input_node_shared_ptr(1);
    if (!reshape || !ov::is_type<ov::op::v1::Reshape>(reshape)) {
        return false;
    }
    auto tile = reshape->get_input_node_shared_ptr(0);
    if (!tile || !ov::is_type<ov::op::v0::Tile>(tile)) {
        return false;
    }
    auto convert = tile->get_input_node_shared_ptr(0);
    if (!convert || !ov::is_type<ov::op::v0::Convert>(convert)) {
        return false;
    }
    auto mask_param = convert->get_input_node_shared_ptr(0);
    if (!mask_param || !ov::is_type<ov::op::v0::Parameter>(mask_param)) {
        return false;
    }
    return mask_param->get_friendly_name().find("attention_mask_global") != std::string::npos;
}

// Pattern nodes for the global-attention decomposed SDPA sub-graph shared by
// QuantizedSDPAWithGlobalMask and SeparateKVCache. The Add node carries the consumes_global_mask
// predicate, so this pattern only matches global-attention blocks -- which is exactly
// what SeparateKVCache needs, since local attention is not isolated/folded.
struct QuantizedSDPAWithGlobalMaskNodes {
    std::shared_ptr<ov::Node> past_k, concat1, convert1, multiply1, transpose1, matmul1;
    std::shared_ptr<ov::Node> add, softmax;
    std::shared_ptr<ov::Node> past_v, concat2, convert2, multiply2;
    std::shared_ptr<ov::Node> matmul2, reshape1, transpose, reshape2, fake_quantize;
};

inline QuantizedSDPAWithGlobalMaskNodes make_sdpa_decomposed1_pattern() {
    QuantizedSDPAWithGlobalMaskNodes n;
    n.past_k = opp::wrap_type<ov::op::v0::Parameter>();
    n.concat1 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), n.past_k});
    n.convert1 = opp::wrap_type<ov::op::v0::Convert>({n.concat1});
    n.multiply1 = opp::wrap_type<ov::op::v1::Multiply>({n.convert1, opp::any_input()});
    n.transpose1 = opp::wrap_type<ov::op::v1::Transpose>({n.multiply1, opp::any_input()});
    n.matmul1 = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), n.transpose1});
    n.add = opp::wrap_type<ov::op::v1::Add>({n.matmul1, opp::any_input()}, consumes_global_mask);
    n.softmax = opp::wrap_type<ov::op::v8::Softmax>({n.add});
    n.past_v = opp::wrap_type<ov::op::v0::Parameter>();
    n.concat2 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), n.past_v});
    n.convert2 = opp::wrap_type<ov::op::v0::Convert>({n.concat2});
    n.multiply2 = opp::wrap_type<ov::op::v1::Multiply>({n.convert2, opp::any_input()});
    n.matmul2 = opp::wrap_type<ov::op::v0::MatMul>({n.softmax, n.multiply2});
    n.reshape1 = opp::wrap_type<ov::op::v1::Reshape>({n.matmul2, opp::any_input()});
    n.transpose = opp::wrap_type<ov::op::v1::Transpose>({n.reshape1, opp::any_input()});
    n.reshape2 = opp::wrap_type<ov::op::v1::Reshape>({n.transpose, opp::any_input()});

    return n;
}
}  // namespace

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

                \       /
                 Concat
                    |
                opt:Unsqueeze
                    |
                opt:Broadcast
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
    // KV-cache Concat predicate: all inputs except the last (current KV slice) must be
    // Parameters or Parameter-through-Convert.  This guards against matching Eagle-style
    // concats or DQ-chain concats whose non-last inputs are intermediate nodes such as
    // Multiply/Subtract produced by a dequantisation path.
    auto kv_concat_pred = [](const ov::Output<ov::Node>& output) {
        auto concat = output.get_node_shared_ptr();
        const auto num_inputs = concat->get_input_size();
        if (num_inputs < 2)
            return false;
        for (size_t i = 0; i + 1 < num_inputs; ++i) {
            auto inp = concat->get_input_node_shared_ptr(i);
            if (ov::as_type_ptr<ov::op::v0::Parameter>(inp))
                continue;
            if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(inp)) {
                if (ov::as_type_ptr<ov::op::v0::Parameter>(cvt->get_input_node_shared_ptr(0)))
                    continue;
            }
            return false;
        }
        return true;
    };

    auto concat1 = opp::wrap_type<ov::op::v0::Concat>(kv_concat_pred);

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

    auto concat2 = opp::wrap_type<ov::op::v0::Concat>(kv_concat_pred);

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

        // Isolate concat nodes and their Convert inputs (if any)
        auto isolate_concat_with_inputs = [&](const auto& concat_pattern) {
            auto concat_iter = node_to_output.find(concat_pattern);
            if (concat_iter != node_to_output.end()) {
                auto concat_node = concat_iter->second.get_node_shared_ptr();
                node_to_gptr->at(concat_node)->isolate(isol_tag);

                // Also isolate all Convert inputs to this Concat
                for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
                    auto input_node = concat_node->get_input_node_shared_ptr(i);
                    if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(input_node)) {
                        if (node_to_gptr->count(convert_node)) {
                            node_to_gptr->at(convert_node)->isolate(isol_tag);
                        }
                    }
                }
            }
        };

        // Isolate Concat nodes with all their Convert inputs
        isolate_concat_with_inputs(concat1);
        isolate_concat_with_inputs(concat2);

        // Isolate all other matched nodes in the pattern
        isolate_matched(unsqueeze1);
        isolate_matched(broadcast1);
        isolate_matched(reshape1);

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

/*
    Decomposed SDPA Pattern with Dynamic Dequantization (i8 KV cache):

    After ConvertKVCacheToPrecision(i8), past KV cache inputs have DQ nodes:
        [any_input] → Subtract(zp) → Multiply(scale) → Concat
    instead of:
        Convert → Concat

    Full pattern:
          Convert
            |
        opt:Subtract (zp)
                |
            Multiply (scale)
                \       /
                 Concat
                    |
                opt:Unsqueeze
                    |
                opt:Broadcast Convert
                    |       \   |
                opt:Reshape opt:Subtract (zp)
                    |            |
                    |        Multiply (scale)
                    |            \       /
            \           /                 Concat
                MatMul                       |
        \       /                      opt:Unsqueeze
           Add                              |
            |                          opt:Broadcast
         Softmax                            |
                \                      opt:Reshape
                    \               /
                          MatMul
                            |
                        Transpose
                            |
                        Reshape
                            |
*/

SDPACompressed::SDPACompressed(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                               const std::string& isol_tag) {
    // Key path: opt:Convert → opt:Subtract(opt:Convert(any), any) → Multiply(any) → Concat(any)
    // Convert is optional to handle models where the past KV is already in the expected type.
    // Subtract is optional to handle both asymmetric (with zp) and symmetric (without zp) DQ.
    // The zp input also has an optional Convert (i8→f32) that must be isolated
    // to preserve the original DQ zp parameter name for PyramidAttention::from().
    auto convert1 = opp::optional<ov::op::v0::Convert>({opp::any_input()});
    auto zp_convert1 = opp::optional<ov::op::v0::Convert>({opp::any_input()});
    auto subtract1 = opp::optional<ov::op::v1::Subtract>({convert1, zp_convert1});
    auto multiply1 = opp::wrap_type<ov::op::v1::Multiply>({subtract1, opp::any_input()});
    auto concat1 = opp::wrap_type<ov::op::v0::Concat>({multiply1, opp::any_input()});

    // GQA optional nodes — single consumer guard
    auto single_user = [](const ov::Output<ov::Node>& output) {
        return output.get_target_inputs().size() == 1;
    };
    auto unsqueeze1 = opp::optional<ov::op::v0::Unsqueeze>({concat1, opp::any_input()}, single_user);
    auto broadcast1 = opp::optional<ov::op::v3::Broadcast>({unsqueeze1, opp::any_input()}, single_user);
    auto reshape1 = opp::optional<ov::op::v1::Reshape>({broadcast1, opp::any_input()}, single_user);

    // Value path: opt:Convert → opt:Subtract(opt:Convert(any), any) → Multiply(any) → Concat(any)
    auto convert2 = opp::optional<ov::op::v0::Convert>({opp::any_input()});
    auto zp_convert2 = opp::optional<ov::op::v0::Convert>({opp::any_input()});
    auto subtract2 = opp::optional<ov::op::v1::Subtract>({convert2, zp_convert2});
    auto multiply2 = opp::wrap_type<ov::op::v1::Multiply>({subtract2, opp::any_input()});
    auto concat2 = opp::wrap_type<ov::op::v0::Concat>({multiply2, opp::any_input()});

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

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        LOG_DEBUG("Decomposed SDPA DQ pattern matched!");

        auto& node_to_output = m.get_pattern_value_map();

        auto isolate_matched = [&](const auto& pattern) {
            auto optional_node = node_to_output.find(pattern);
            if (optional_node != node_to_output.end()) {
                auto matched_node = optional_node->second.get_node_shared_ptr();
                node_to_gptr->at(matched_node)->isolate(isol_tag);
            }
        };

        // Isolate all matched nodes in the pattern
        isolate_matched(convert1);
        isolate_matched(zp_convert1);
        isolate_matched(subtract1);
        isolate_matched(multiply1);
        isolate_matched(concat1);
        isolate_matched(unsqueeze1);
        isolate_matched(broadcast1);
        isolate_matched(reshape1);

        isolate_matched(convert2);
        isolate_matched(zp_convert2);
        isolate_matched(subtract2);
        isolate_matched(multiply2);
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

    register_matcher(std::make_shared<opp::Matcher>(reshape3, "TagSDPADecomposedDQ"), std::move(callback));
}

QuantizedSDPAWithGlobalMask::QuantizedSDPAWithGlobalMask(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                                         const std::string& isol_tag) {
    // AttentionBroadcast4 pre-folds the shape sub-graph before this pass runs.
    auto n = make_sdpa_decomposed1_pattern();

    auto node_to_gptr = snapshot->getNodeToGroupMap();

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

        auto concat_name = node_to_output.at(n.concat1).get_node()->get_friendly_name();
        int block_index = 0;
        std::regex pattern(R"(_module\.decoder\.blocks\.(\d+)\..*)");
        std::smatch match;
        if (std::regex_match(concat_name, match, pattern)) {
            block_index = std::stoi(match[1].str());
        } else {
            // doesn't matter
        }

        node_to_output.at(n.past_k).get_node()->set_friendly_name("past_key_values." + std::to_string(block_index) +
                                                                  ".key");
        node_to_output.at(n.past_v).get_node()->set_friendly_name("past_key_values." + std::to_string(block_index) +
                                                                  ".value");

        isolate_matched(n.concat1);
        isolate_matched(n.convert1);
        isolate_matched(n.multiply1);
        isolate_matched(n.transpose1);
        isolate_matched(n.matmul1);

        isolate_matched(n.add);

        isolate_matched(n.softmax);

        isolate_matched(n.concat2);
        isolate_matched(n.convert2);
        isolate_matched(n.multiply2);

        isolate_matched(n.matmul2);
        isolate_matched(n.reshape1);
        isolate_matched(n.transpose);
        isolate_matched(n.reshape2);

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(n.reshape2, "TagQuantizedSDPAWithGlobalMask"), std::move(callback));
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
    auto param_cvt = opp::optional<ov::op::v0::Convert>({param_in});
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
// appears on the attention mask path (originally part of QuantizedSDPAWithGlobalMask before
// it was simplified). Running this before QuantizedSDPAWithGlobalMask allows that pattern
// to use any_input() for the Add's second operand.
AttentionBroadcast4::AttentionBroadcast4() {
    // Pattern derived from the original QuantizedSDPAWithGlobalMask mask-shape sub-graph:
    //   ShapeOf(kv) -> Gather
    //                        Concat(any, any, any, Gather) -> Reshape -> Add
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>(multiply);
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_gather =
        opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input(), opp::any_input(), gather});
    auto reshape_gather = opp::wrap_type<ov::op::v1::Reshape>({opp::any_input(), concat_gather});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_gather_out = node_to_output.at(gather);
        if (matched_gather_out.get_target_inputs().size() > 1) {
            // This pattern is for the Gather that feeds a single Concat.
            return false;
        }
        auto matched_concat_out = node_to_output.at(concat_gather);
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

            return true;  // root changed
        }
        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape_gather, "AttentionBroadcast4"), std::move(callback));
}

// SeparateKVCache: OpenVINO's SharedOpOptimization (run by Model::reshape) merges the
// identical K- and V-cache dequantization chains across global-attention blocks that
// reuse the same KV cache, so a single K-transpose / V-multiply can feed several blocks'
// MatMuls. The NPUW repeated-block partitioner cannot represent such a shared node -- it
// leaks as an extra output of one block while being consumed by others, producing repeated
// blocks with inconsistent output counts. This pass duplicates the shared cache chain per
// consumer so each attention block owns an independent copy. The K side
// (Concat->Convert->Multiply->Transpose) and the V side (Concat->Convert->Multiply) are
// handled the same way.
//
// NB: This pass reuses the QuantizedSDPAWithGlobalMask pattern (with the consumes_global_mask
// predicate), so it only fires for global-attention blocks. Local attention is not
// isolated/folded here, so leaving its (also shared) KV chains untouched is fine.
SeparateKVCache::SeparateKVCache() {
    auto n = make_sdpa_decomposed1_pattern();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        // Clone a (possibly weight) constant so each duplicated chain owns its own node.
        // Sharing one constant across multiple call-site subgraph models confuses the
        // partitioner's propagateWeights bank-assignment, which expects exactly one
        // constant per call site. Non-constant sources (e.g. a Parameter) are reused.
        auto clone_source = [](const ov::Output<ov::Node>& src) -> ov::Output<ov::Node> {
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(src.get_node_shared_ptr())) {
                return std::make_shared<ov::op::v0::Constant>(*c)->output(0);
            }
            return src;
        };

        // Duplicate the cache chain feeding `shared_node` for every consumer other than the
        // one already matched in this pattern. `clone_chain` builds a fresh, independent copy
        // of the chain. Returns true if any consumer was redirected.
        auto unshare = [&](const std::shared_ptr<ov::Node>& shared_node,
                           const ov::Node* pattern_consumer,
                           auto&& clone_chain) -> bool {
            if (!shared_node) {
                return false;
            }
            auto targets = shared_node->output(0).get_target_inputs();
            if (targets.size() <= 1) {
                return false;  // only the matched consumer -- nothing to unshare.
            }
            bool changed = false;
            for (auto& target_input : targets) {
                if (target_input.get_node() == pattern_consumer) {
                    continue;  // keep the matched consumer on the original chain.
                }
                if (!dynamic_cast<ov::op::v0::MatMul*>(target_input.get_node())) {
                    continue;
                }
                target_input.replace_source_output(clone_chain());
                changed = true;
            }
            return changed;
        };

        auto concat1_node =
            std::dynamic_pointer_cast<ov::op::v0::Concat>(node_to_output.at(n.concat1).get_node_shared_ptr());
        auto convert1_node =
            std::dynamic_pointer_cast<ov::op::v0::Convert>(node_to_output.at(n.convert1).get_node_shared_ptr());
        auto multiply1_node =
            std::dynamic_pointer_cast<ov::op::v1::Multiply>(node_to_output.at(n.multiply1).get_node_shared_ptr());
        auto transpose1_node =
            std::dynamic_pointer_cast<ov::op::v1::Transpose>(node_to_output.at(n.transpose1).get_node_shared_ptr());
        auto matmul1_node =
            std::dynamic_pointer_cast<ov::op::v0::MatMul>(node_to_output.at(n.matmul1).get_node_shared_ptr());

        auto concat2_node =
            std::dynamic_pointer_cast<ov::op::v0::Concat>(node_to_output.at(n.concat2).get_node_shared_ptr());
        auto convert2_node =
            std::dynamic_pointer_cast<ov::op::v0::Convert>(node_to_output.at(n.convert2).get_node_shared_ptr());
        auto multiply2_node =
            std::dynamic_pointer_cast<ov::op::v1::Multiply>(node_to_output.at(n.multiply2).get_node_shared_ptr());
        auto matmul2_node =
            std::dynamic_pointer_cast<ov::op::v0::MatMul>(node_to_output.at(n.matmul2).get_node_shared_ptr());

        auto concat1_inputs = concat1_node->inputs();
        auto multiply1_inputs = multiply1_node->inputs();
        auto transpose1_inputs = transpose1_node->inputs();
        auto concat2_inputs = concat2_node->inputs();
        auto multiply2_inputs = multiply2_node->inputs();

        // K-cache chain clone: Concat -> Convert -> Multiply -> Transpose
        auto clone_k = [&]() -> ov::Output<ov::Node> {
            auto new_concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{concat1_inputs[0].get_source_output(), concat1_inputs[1].get_source_output()},
                concat1_node->get_axis());
            auto new_convert =
                std::make_shared<ov::op::v0::Convert>(new_concat, convert1_node->get_convert_element_type());
            auto new_multiply =
                std::make_shared<ov::op::v1::Multiply>(new_convert,
                                                       clone_source(multiply1_inputs[1].get_source_output()));
            auto new_transpose =
                std::make_shared<ov::op::v1::Transpose>(new_multiply,
                                                        clone_source(transpose1_inputs[1].get_source_output()));
            return new_transpose->output(0);
        };

        // V-cache chain clone: Concat -> Convert -> Multiply
        auto clone_v = [&]() -> ov::Output<ov::Node> {
            auto new_concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{concat2_inputs[0].get_source_output(), concat2_inputs[1].get_source_output()},
                concat2_node->get_axis());
            auto new_convert =
                std::make_shared<ov::op::v0::Convert>(new_concat, convert2_node->get_convert_element_type());
            auto new_multiply =
                std::make_shared<ov::op::v1::Multiply>(new_convert,
                                                       clone_source(multiply2_inputs[1].get_source_output()));
            return new_multiply->output(0);
        };

        bool changed = false;
        changed = unshare(transpose1_node, matmul1_node.get(), clone_k) || changed;
        changed = unshare(multiply2_node, matmul2_node.get(), clone_v) || changed;
        return changed;
    };

    register_matcher(std::make_shared<opp::Matcher>(n.reshape2, "SeparateKVCache"), std::move(callback));
}

bool RegularizeSDPA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool model_changed = false;
    if (m_run_broadcast_pattern) {
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast2>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast3>();
        rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast4>();
        rewr.add_matcher<ov::npuw::patterns::regularize::SeparateKVCache>();

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
