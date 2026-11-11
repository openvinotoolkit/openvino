// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mask_linear_attention_padding.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "right_align_mask_slice_for_conv.hpp"

namespace {

namespace opv0 = ov::op::v0;
namespace opv1 = ov::op::v1;
namespace opv8 = ov::op::v8;

constexpr size_t kSeqAxis = 2;

struct ConvSite {
    std::shared_ptr<opv0::Concat> concat;    // Concat(past [B,C,K], x [B,C,L]) along the sequence axis
    size_t kernel;                           // K, the width of the carried history
    size_t seq_len;                          // L, the number of tokens of the chunk
    std::shared_ptr<opv0::Parameter> state;  // the past conv state parameter the history comes from
};

struct SsmSite {
    std::shared_ptr<ov::op::v5::Loop> loop;
    size_t gate_input;                       // Loop input index of the per-token decay g
    size_t beta_input;                       // Loop input index of the per-token update gate beta
    size_t seq_len;                          // L, the number of tokens of the chunk
    std::shared_ptr<opv0::Parameter> state;  // the past ssm state parameter the recurrence starts from
};

bool has_name(const ov::descriptor::Tensor& tensor, const char* kind, const std::string& past_or_present) {
    const std::string needle = std::string(".") + kind + ".";
    return std::any_of(tensor.get_names().begin(), tensor.get_names().end(), [&](const std::string& name) {
        return name.find(needle) != std::string::npos && ov::npuw::util::matchLinCacheString(name, past_or_present);
    });
}

bool is_past_state(const ov::Output<ov::Node>& output, const char* kind) {
    const auto param = ov::as_type_ptr<opv0::Parameter>(output.get_node_shared_ptr());
    return param && has_name(param->get_output_tensor(0), kind, "past");
}

// Concat(past conv state, x) along the last axis. The state may reach the Concat directly or through a
// Transpose which re-lays it out.
std::optional<ConvSite> match_conv(const std::shared_ptr<ov::Node>& node) {
    const auto concat = ov::as_type_ptr<opv0::Concat>(node);
    if (!concat || concat->get_input_size() != 2) {
        return std::nullopt;
    }
    const auto& out_shape = concat->get_output_partial_shape(0);
    if (out_shape.rank().is_dynamic() || out_shape.size() != 3 ||
        ov::util::normalize_axis(concat->get_axis(), 3) != static_cast<int64_t>(kSeqAxis)) {
        return std::nullopt;
    }

    auto state = concat->input_value(0);
    if (const auto transpose = ov::as_type_ptr<opv1::Transpose>(state.get_node_shared_ptr())) {
        state = transpose->input_value(0);
    }
    if (!is_past_state(state, "conv")) {
        return std::nullopt;
    }
    const auto state_param = ov::as_type_ptr<opv0::Parameter>(state.get_node_shared_ptr());

    // The output rank does not prove the input ranks: one rank-dynamic input is enough to infer it.
    const auto& history = concat->get_input_partial_shape(0);
    const auto& tokens = concat->get_input_partial_shape(1);
    for (const auto& shape : {history, tokens}) {
        if (shape.rank().is_dynamic() || shape.size() != 3 || shape[kSeqAxis].is_dynamic()) {
            return std::nullopt;
        }
    }
    return ConvSite{concat,
                    static_cast<size_t>(history[kSeqAxis].get_length()),
                    static_cast<size_t>(tokens[kSeqAxis].get_length()),
                    state_param};
}

// The only consumer of `output`, when it is of type T.
template <class T>
std::shared_ptr<ov::Node> sole_consumer(const ov::Output<ov::Node>& output) {
    const auto targets = output.get_target_inputs();
    if (targets.size() != 1 || !ov::is_type<T>(targets.begin()->get_node())) {
        return nullptr;
    }
    return targets.begin()->get_node()->shared_from_this();
}

// Skips an optional Convert (e.g. f16 -> f32) that is the only consumer of the node.
std::shared_ptr<ov::Node> skip_convert(const std::shared_ptr<ov::Node>& node) {
    if (const auto convert = sole_consumer<opv0::Convert>(node->output(0))) {
        return convert;
    }
    return node;
}

// Skips an optional Convert that produces the value, looking at the node that feeds it.
std::shared_ptr<ov::Node> skip_convert_producer(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<opv0::Convert>(node) ? node->input_value(0).get_node_shared_ptr() : node;
}

// g: [Convert] -> Exp -> Unsqueeze -> Multiply(state, .). Returns that Multiply, the decayed state.
std::shared_ptr<ov::Node> match_gate(const std::shared_ptr<opv0::Parameter>& gate) {
    auto node = sole_consumer<opv0::Exp>(skip_convert(gate)->output(0));
    node = node ? sole_consumer<opv0::Unsqueeze>(node->output(0)) : nullptr;
    return node ? sole_consumer<opv1::Multiply>(node->output(0)) : nullptr;
}

// beta: [Convert] -> Multiply(delta, .) -> Unsqueeze -> Multiply(key, .) -> Add(decayed_state, .).
// Returns that Add, the updated state.
std::shared_ptr<ov::Node> match_beta(const std::shared_ptr<opv0::Parameter>& beta,
                                     const std::shared_ptr<ov::Node>& decayed_state) {
    auto node = sole_consumer<opv1::Multiply>(skip_convert(beta)->output(0));
    node = node ? sole_consumer<opv0::Unsqueeze>(node->output(0)) : nullptr;
    node = node ? sole_consumer<opv1::Multiply>(node->output(0)) : nullptr;
    node = node ? sole_consumer<opv1::Add>(node->output(0)) : nullptr;
    const bool adds_to_decayed_state = node && (node->input_value(0).get_node() == decayed_state.get() ||
                                                node->input_value(1).get_node() == decayed_state.get());
    return adds_to_decayed_state ? node : nullptr;
}

// The recurrence Loop of GatedDeltaNet: q, k, v are sliced as [B,H,L,D] and the two per-token scalars
// g and beta as [B,H,L]. g and beta are identified by the body, see match_gate and match_beta.
std::optional<SsmSite> match_ssm(const std::shared_ptr<ov::Node>& node) {
    const auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop) {
        return std::nullopt;
    }
    const auto body = loop->get_function();
    std::vector<std::pair<size_t, std::shared_ptr<opv0::Parameter>>> per_token_scalars;
    for (const auto& desc : loop->get_input_descriptions()) {
        const auto slice = ov::as_type_ptr<ov::op::util::SubGraphOp::SliceInputDescription>(desc);
        if (!slice || slice->m_axis != static_cast<int64_t>(kSeqAxis)) {
            continue;
        }
        const auto& shape = loop->get_input_partial_shape(slice->m_input_index);
        if (shape.rank().is_dynamic() || shape.size() != 3) {
            continue;
        }
        per_token_scalars.emplace_back(slice->m_input_index, body->get_parameters().at(slice->m_body_parameter_index));
    }
    if (per_token_scalars.size() != 2) {
        return std::nullopt;
    }

    // The scalars are told apart by how the body uses them, not by elimination: g decays the state and beta
    // scales the delta that is added to the decayed state, so a zero g / beta is an identity step.
    size_t gate = 0;
    size_t beta = 0;
    std::shared_ptr<ov::Node> decayed_state;
    std::shared_ptr<ov::Node> updated_state;
    bool found = false;
    for (const auto& [gate_candidate, beta_candidate] : {std::pair{0u, 1u}, std::pair{1u, 0u}}) {
        decayed_state = match_gate(per_token_scalars[gate_candidate].second);
        updated_state = decayed_state ? match_beta(per_token_scalars[beta_candidate].second, decayed_state) : nullptr;
        if (updated_state) {
            gate = per_token_scalars[gate_candidate].first;
            beta = per_token_scalars[beta_candidate].first;
            found = true;
            break;
        }
    }
    if (!found) {
        return std::nullopt;
    }
    const auto& seq_dim = loop->get_input_partial_shape(gate)[kSeqAxis];
    if (seq_dim.is_dynamic()) {
        return std::nullopt;
    }

    std::shared_ptr<opv0::Parameter> state;
    std::shared_ptr<opv0::Parameter> state_in;
    std::shared_ptr<ov::op::util::SubGraphOp::MergedInputDescription> state_merge;
    for (const auto& desc : loop->get_input_descriptions()) {
        auto source = loop->input_value(desc->m_input_index);
        if (ov::is_type<opv0::Convert>(source.get_node())) {
            source = source.get_node()->input_value(0);
        }
        if (is_past_state(source, "ssm")) {
            if (state) {
                return std::nullopt;
            }
            state = ov::as_type_ptr<opv0::Parameter>(source.get_node_shared_ptr());
            state_in = body->get_parameters().at(desc->m_body_parameter_index);
            state_merge = ov::as_type_ptr<ov::op::util::SubGraphOp::MergedInputDescription>(desc);
        }
    }
    if (!state || !state_merge) {
        return std::nullopt;
    }

    // Nothing else updates the state after the delta is added: the Add (optionally converted back) is
    // what the next iteration starts from.
    const auto carried = body->get_results().at(state_merge->m_body_value_index)->input_value(0).get_node_shared_ptr();
    if (skip_convert_producer(carried) != updated_state) {
        return std::nullopt;
    }

    // The decay is applied to the state the recurrence carries, not to some other tensor.
    const auto is_state_in = [&](const ov::Output<ov::Node>& value) {
        const auto node = ov::is_type<opv0::Convert>(value.get_node()) ? value.get_node()->input_value(0).get_node()
                                                                       : value.get_node();
        return node == state_in.get();
    };
    if (!is_state_in(decayed_state->input_value(0)) && !is_state_in(decayed_state->input_value(1))) {
        return std::nullopt;
    }
    return SsmSite{loop, gate, beta, static_cast<size_t>(seq_dim.get_length()), state};
}

// True when the sites together consume every past state parameter of the given kind.
template <class Sites>
bool covers_all_states(const std::shared_ptr<ov::Model>& model, const char* kind, const Sites& sites) {
    std::set<const ov::Node*> covered;
    for (const auto& site : sites) {
        covered.insert(site.state.get());
    }
    for (const auto& param : model->get_parameters()) {
        if (has_name(param->get_output_tensor(0), kind, "past") && covered.count(param.get()) == 0) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<opv0::Parameter> find_attention_mask(const std::shared_ptr<ov::Model>& model) {
    for (const auto& param : model->get_parameters()) {
        const auto& names = param->get_output_tensor(0).get_names();
        if (names.count(ov::npuw::attention_mask_name) != 0 ||
            param->get_friendly_name() == ov::npuw::attention_mask_name) {
            return param;
        }
    }
    return nullptr;
}

std::shared_ptr<opv0::Constant> i64_const(const std::vector<int64_t>& values, const ov::Shape& shape) {
    return opv0::Constant::create(ov::element::i64, shape, values);
}

std::shared_ptr<opv0::Constant> i64_scalar(int64_t value) {
    return i64_const({value}, ov::Shape{});
}

std::shared_ptr<opv0::Constant> i64_vector(int64_t value) {
    return i64_const({value}, ov::Shape{1});
}

// Columns of the conv input that receive the history: pad + [0, K). The real tokens of the chunk start at
// column K + pad of the Concat, so writing the history there puts it right in front of them:
//   [h0 h1 h2 h3 | 0 0 x2 x3] -> [h0 h1 h0 h1 h2 h3 x2 x3]  (K = 4, L = 4, pad = 2).
// The leading `pad` columns keep whatever they held: they only reach the convolution outputs of the padded
// positions, which are identity steps for the ssm state and are masked out everywhere else.
ov::Output<ov::Node> make_history_columns(const ov::Output<ov::Node>& pad, size_t kernel) {
    std::vector<int64_t> offsets(kernel);
    for (size_t i = 0; i < kernel; ++i) {
        offsets[i] = static_cast<int64_t>(i);
    }
    return std::make_shared<opv1::Add>(pad, i64_const(offsets, ov::Shape{kernel}))->output(0);
}

}  // namespace

bool ov::npuw::MaskLinearAttentionPadding::run_on_model(const std::shared_ptr<ov::Model>& model) {
    size_t past_conv_count = 0;
    size_t past_ssm_count = 0;
    for (const auto& param : model->get_parameters()) {
        past_conv_count += has_name(param->get_output_tensor(0), "conv", "past") ? 1u : 0u;
        past_ssm_count += has_name(param->get_output_tensor(0), "ssm", "past") ? 1u : 0u;
    }
    if (past_conv_count == 0 && past_ssm_count == 0) {
        return false;
    }

    std::vector<ConvSite> conv_sites;
    std::vector<SsmSite> ssm_sites;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto conv = match_conv(node)) {
            conv_sites.push_back(*conv);
        } else if (auto ssm = match_ssm(node)) {
            ssm_sites.push_back(*ssm);
        }
    }
    if (!covers_all_states(model, "conv", conv_sites) || !covers_all_states(model, "ssm", ssm_sites)) {
        LOG_WARN("MaskLinearAttentionPadding: "
                 << conv_sites.size() << " conv and " << ssm_sites.size() << " ssm sites do not cover all "
                 << past_conv_count << " conv and " << past_ssm_count
                 << " ssm states, model left unchanged: a partial prefill chunk may corrupt them");
        return false;
    }

    // The sequence length of the chunk must be the same for all states.
    std::optional<size_t> seq_len;
    auto accept_seq_len = [&](size_t length) {
        if (seq_len && *seq_len != length) {
            return false;
        }
        seq_len = length;
        return true;
    };
    for (const auto& site : conv_sites) {
        if (!accept_seq_len(site.seq_len)) {
            LOG_WARN("MaskLinearAttentionPadding: conv sequence length differs between layers, model left unchanged: a "
                     "partial prefill chunk may corrupt the states");
            return false;
        }
    }
    for (const auto& site : ssm_sites) {
        if (!accept_seq_len(site.seq_len)) {
            LOG_WARN("MaskLinearAttentionPadding: ssm sequence length differs between layers, model left unchanged: a "
                     "partial prefill chunk may corrupt the states");
            return false;
        }
    }

    const auto attention_mask = find_attention_mask(model);
    if (!attention_mask || attention_mask->get_partial_shape().is_dynamic() ||
        attention_mask->get_partial_shape().size() != 2 ||
        static_cast<size_t>(attention_mask->get_partial_shape()[1].get_length()) < *seq_len) {
        LOG_WARN(
            "MaskLinearAttentionPadding: static rank-2 attention_mask is not found, model left unchanged: a partial "
            "prefill chunk may corrupt the states");
        return false;
    }

    // Mask of the chunk: the last L columns of the attention mask. The padding is at its beginning.
    const int64_t total = attention_mask->get_partial_shape()[1].get_length();
    const int64_t len = static_cast<int64_t>(*seq_len);
    const auto chunk_mask = std::make_shared<opv8::Slice>(attention_mask,
                                                          i64_vector(total - len),
                                                          i64_vector(total),
                                                          i64_vector(1),
                                                          i64_vector(1));
    chunk_mask->set_friendly_name("lincache_chunk_mask");
    const auto valid_tokens = std::make_shared<opv1::ReduceSum>(chunk_mask, i64_const({0, 1}, ov::Shape{2}), false);
    const auto pad = std::make_shared<opv1::Subtract>(i64_scalar(len), valid_tokens);

    std::map<size_t, ov::Output<ov::Node>> columns_by_kernel;
    const auto seq_axis = i64_scalar(static_cast<int64_t>(kSeqAxis));
    for (const auto& site : conv_sites) {
        auto columns_it = columns_by_kernel.find(site.kernel);
        if (columns_it == columns_by_kernel.end()) {
            columns_it = columns_by_kernel.emplace(site.kernel, make_history_columns(pad, site.kernel)).first;
        }

        // The history [B, C, K] is written over K columns of the Concat, nothing else of it is touched.
        const auto consumers = site.concat->output(0).get_target_inputs();
        const auto reordered = std::make_shared<ov::op::v3::ScatterUpdate>(site.concat,
                                                                           columns_it->second,
                                                                           site.concat->input_value(0),
                                                                           seq_axis);
        reordered->set_friendly_name(site.concat->get_friendly_name() + "/pad_aware");
        for (auto consumer : consumers) {
            consumer.replace_source_output(reordered);
        }
    }

    if (!ssm_sites.empty()) {
        // [B, 1, L], broadcast over the heads of the per-token [B, H, L] scalars.
        const auto mask_per_token = std::make_shared<opv0::Unsqueeze>(chunk_mask, i64_vector(1));
        // One mask per precision, shared by all the gates and betas of that precision.
        std::map<ov::element::Type_t, ov::Output<ov::Node>> mask_by_type;
        for (const auto& site : ssm_sites) {
            for (const auto input_idx : {site.gate_input, site.beta_input}) {
                const auto source = site.loop->input_value(input_idx);
                const auto type = source.get_element_type();
                auto mask_it = mask_by_type.find(type);
                if (mask_it == mask_by_type.end()) {
                    ov::Output<ov::Node> mask = mask_per_token;
                    if (type != mask.get_element_type()) {
                        mask = std::make_shared<opv0::Convert>(mask, type);
                    }
                    mask_it = mask_by_type.emplace(type, mask).first;
                }
                const auto masked = std::make_shared<opv1::Multiply>(source, mask_it->second);
                site.loop->input(input_idx).replace_source_output(masked);
            }
        }
    }

    model->validate_nodes_and_infer_types();
    LOG_DEBUG("MaskLinearAttentionPadding: " << conv_sites.size() << " conv and " << ssm_sites.size()
                                             << " ssm states are padding-safe (chunk length " << len << ")");
    return true;
}
