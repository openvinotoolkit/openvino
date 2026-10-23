// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mask_linear_attention_padding.hpp"

#include <algorithm>
#include <map>
#include <optional>
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
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
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
    std::shared_ptr<opv0::Concat> concat;  // Concat(past [B,C,K], x [B,C,L]) along the sequence axis
    size_t kernel;                         // K, the width of the carried history
};

struct SsmSite {
    std::shared_ptr<ov::op::v5::Loop> loop;
    size_t gate_input;  // Loop input index of the per-token decay g
    size_t beta_input;  // Loop input index of the per-token update gate beta
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

    const auto& history = concat->get_input_partial_shape(0);
    const auto& tokens = concat->get_input_partial_shape(1);
    if (history[kSeqAxis].is_dynamic() || tokens[kSeqAxis].is_dynamic()) {
        return std::nullopt;
    }
    return ConvSite{concat, static_cast<size_t>(history[kSeqAxis].get_length())};
}

// The recurrence Loop of GatedDeltaNet: q, k, v are sliced as [B,H,L,D] and the two per-token scalars
// g and beta as [B,H,L]. g is the one that goes through Exp inside the body.
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

    auto feeds_exp = [](const std::shared_ptr<opv0::Parameter>& param) {
        const auto targets = param->output(0).get_target_inputs();
        return std::any_of(targets.begin(), targets.end(), [](const ov::Input<ov::Node>& target) {
            return ov::is_type<opv0::Exp>(target.get_node());
        });
    };
    const bool first_is_gate = feeds_exp(per_token_scalars[0].second);
    const bool second_is_gate = feeds_exp(per_token_scalars[1].second);
    if (first_is_gate == second_is_gate) {
        return std::nullopt;
    }
    const size_t gate = first_is_gate ? per_token_scalars[0].first : per_token_scalars[1].first;
    const size_t beta = first_is_gate ? per_token_scalars[1].first : per_token_scalars[0].first;
    return SsmSite{loop, gate, beta};
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

// Shared by every conv state with the same history width and precision.
struct ConvReorder {
    ov::Output<ov::Node> shift;            // [K, K+L] one-hot, shift[i][j] = (j == pad + i)
    ov::Output<ov::Node> in_shifted_part;  // [1,1,K+L] true for the first pad + K columns
};

// The reordered conv input is  [0 x pad | history (K) | real tokens]  of the same width K+L as the Concat.
// Its first pad + K columns come from the history shifted right by `pad` (all zeros for the first
// `pad` columns), computed as history[B,C,K] x one-hot[K,K+L] to avoid a data-dependent Gather;
// the remaining columns are those of the Concat, which already holds the real tokens at the right place.
ConvReorder make_conv_reorder(const ov::Output<ov::Node>& pad,
                              size_t kernel,
                              size_t seq_len,
                              const ov::element::Type& type) {
    const size_t width = kernel + seq_len;
    std::vector<int64_t> columns(width);
    for (size_t j = 0; j < width; ++j) {
        columns[j] = static_cast<int64_t>(j);
    }
    std::vector<int64_t> rows(kernel);
    for (size_t i = 0; i < kernel; ++i) {
        rows[i] = static_cast<int64_t>(i);
    }
    const auto column = i64_const(columns, ov::Shape{1, width});
    const auto target = std::make_shared<opv1::Add>(pad, i64_const(rows, ov::Shape{kernel, 1}));
    const auto shift = std::make_shared<opv0::Convert>(std::make_shared<opv1::Equal>(column, target), type);

    const auto history_end = std::make_shared<opv1::Add>(pad, i64_scalar(static_cast<int64_t>(kernel)));
    const auto in_shifted_part = std::make_shared<opv1::Less>(i64_const(columns, ov::Shape{1, 1, width}), history_end);
    return {shift->output(0), in_shifted_part->output(0)};
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
    if (conv_sites.size() != past_conv_count || ssm_sites.size() != past_ssm_count) {
        LOG_WARN("MaskLinearAttentionPadding: recognized " << conv_sites.size() << "/" << past_conv_count
                                                           << " conv and " << ssm_sites.size() << "/" << past_ssm_count
                                                           << " ssm states, model left unchanged");
        return false;
    }

    // The sequence length of the chunk must be the same for all states.
    std::optional<size_t> seq_len;
    auto accept_seq_len = [&](const ov::Dimension& dim) {
        if (dim.is_dynamic() || (seq_len && *seq_len != static_cast<size_t>(dim.get_length()))) {
            return false;
        }
        seq_len = static_cast<size_t>(dim.get_length());
        return true;
    };
    for (const auto& site : conv_sites) {
        if (!accept_seq_len(site.concat->get_input_partial_shape(1)[kSeqAxis])) {
            LOG_WARN("MaskLinearAttentionPadding: conv sequence length is dynamic or differs between layers");
            return false;
        }
    }
    for (const auto& site : ssm_sites) {
        if (!accept_seq_len(site.loop->get_input_partial_shape(site.gate_input)[kSeqAxis])) {
            LOG_WARN("MaskLinearAttentionPadding: ssm sequence length is dynamic or differs between layers");
            return false;
        }
    }

    const auto attention_mask = find_attention_mask(model);
    if (!attention_mask || attention_mask->get_partial_shape().is_dynamic() ||
        attention_mask->get_partial_shape().size() != 2 ||
        static_cast<size_t>(attention_mask->get_partial_shape()[1].get_length()) < *seq_len) {
        LOG_WARN("MaskLinearAttentionPadding: static rank-2 attention_mask is not found, model left unchanged");
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

    std::map<std::pair<size_t, ov::element::Type_t>, ConvReorder> reorder_by_kernel;
    for (const auto& site : conv_sites) {
        const auto type = site.concat->get_output_element_type(0);
        const auto key = std::make_pair(site.kernel, static_cast<ov::element::Type_t>(type));
        auto reorder_it = reorder_by_kernel.find(key);
        if (reorder_it == reorder_by_kernel.end()) {
            reorder_it = reorder_by_kernel.emplace(key, make_conv_reorder(pad, site.kernel, *seq_len, type)).first;
        }
        const auto& reorder = reorder_it->second;

        const auto consumers = site.concat->output(0).get_target_inputs();
        const auto shifted_history = std::make_shared<opv0::MatMul>(site.concat->input_value(0), reorder.shift);
        const auto reordered = std::make_shared<opv1::Select>(reorder.in_shifted_part, shifted_history, site.concat);
        reordered->set_friendly_name(site.concat->get_friendly_name() + "/pad_aware");
        for (auto consumer : consumers) {
            consumer.replace_source_output(reordered);
        }
    }

    if (!ssm_sites.empty()) {
        // [B, 1, L], broadcast over the heads of the per-token [B, H, L] scalars.
        const auto mask_per_token = std::make_shared<opv0::Unsqueeze>(chunk_mask, i64_vector(1));
        for (const auto& site : ssm_sites) {
            for (const auto input_idx : {site.gate_input, site.beta_input}) {
                const auto source = site.loop->input_value(input_idx);
                ov::Output<ov::Node> mask = mask_per_token;
                if (source.get_element_type() != mask.get_element_type()) {
                    mask = std::make_shared<opv0::Convert>(mask, source.get_element_type());
                }
                const auto masked = std::make_shared<opv1::Multiply>(source, mask);
                site.loop->input(input_idx).replace_source_output(masked);
            }
        }
    }

    model->validate_nodes_and_infer_types();
    LOG_DEBUG("MaskLinearAttentionPadding: " << conv_sites.size() << " conv and " << ssm_sites.size()
                                             << " ssm states are padding-safe (chunk length " << len << ")");
    return true;
}
