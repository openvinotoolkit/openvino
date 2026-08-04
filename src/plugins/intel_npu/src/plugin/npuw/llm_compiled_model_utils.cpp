// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include <cstring>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

bool ov::npuw::util::has_input(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto inputs = model->inputs();
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    return it != inputs.end();
}

std::shared_ptr<ov::Node> ov::npuw::util::find_kv_cache_concat(const std::shared_ptr<ov::Node>& sdpa) {
    // The key is SDPA input 1, reached as Concat -> Broadcast -> Reshape -> SDPA when the model
    // carries a GQA KV cache.
    auto* reshape_node = sdpa->input(1).get_source_output().get_node();
    if (reshape_node == nullptr || strstr(reshape_node->get_type_name(), "Reshape") == nullptr) {
        return nullptr;
    }
    auto* broadcast_node = reshape_node->input(0).get_source_output().get_node();
    if (broadcast_node == nullptr || strstr(broadcast_node->get_type_name(), "Broadcast") == nullptr) {
        return nullptr;
    }
    auto* concat_node = broadcast_node->input(1).get_source_output().get_node();
    if (concat_node == nullptr || strstr(concat_node->get_type_name(), "Concat") == nullptr) {
        return nullptr;
    }
    return concat_node->shared_from_this();
}

bool ov::npuw::util::is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    // The autoregressive embedding path reconstructs a prefill/KV model out of the KV-cache
    // concat feeding each SDPA key. A bidirectional encoder (BERT) has SDPA but no such concat,
    // so there is nothing to reconstruct and it has to run as a single forward instead. Routing
    // on the same predicate the reconstruction uses keeps the two from ever disagreeing.
    bool has_sdpa = false;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            has_sdpa = true;
            if (find_kv_cache_concat(op)) {
                // Autoregressive (Qwen3-Embedding-style), handled by PrepareTextEmbeddingModel.
                return false;
            }
        }
    }
    return has_sdpa;
}

void ov::npuw::util::validate_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& param : model->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names()) {
            OPENVINO_ASSERT(name.find("past_key_values") == std::string::npos,
                            "Encoder embedding model has an unexpected autoregressive KV-cache input '",
                            name,
                            "'.");
        }
    }
}

std::optional<uint32_t> ov::npuw::util::get_max_position_embeddings(const std::shared_ptr<ov::Model>& model) {
    namespace opp = ov::pass::pattern;

    // Every embedding table in a BERT-like encoder is read by the same shape of subgraph: a rank-2
    // constant [rows, hidden] behind a Gather, with whatever Convert (and, for compressed weights,
    // Subtract and Multiply) decompression left in place. Spell those intermediates out rather
    // than walking a fixed number of hops through whatever happens to sit above the Gather.
    // int4 and nf4 exports also convert back up to the activation type after scaling, so allow a
    // Convert on either side of the dequantization.
    auto table = opp::wrap_type<ov::op::v0::Constant>();
    auto converted = opp::optional<ov::op::v0::Convert>({table->output(0)});
    auto unzeroed = opp::optional<ov::op::v1::Subtract>({converted->output(0), opp::any_input()});
    auto scaled = opp::optional<ov::op::v1::Multiply>({unzeroed->output(0), opp::any_input()});
    auto upcast = opp::optional<ov::op::v0::Convert>({scaled->output(0)});
    auto lookup = opp::wrap_type<ov::op::v8::Gather>({upcast->output(0), opp::any_input(), opp::any_input()});

    // The word, token-type and position tables are all the same lookup, so the topology on its own
    // cannot tell them apart and the exported name has to do it. It now chooses among candidates
    // the pattern has already established are embedding tables, instead of being the only thing
    // between us and an unrelated Gather. Picking the wrong table here would be expensive: the
    // token-type one has two rows and would clamp the sequence length to nothing.
    opp::Matcher matcher(lookup, "npuw::PositionEmbeddingTable");
    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::v8::Gather>(op) ||
            op->get_friendly_name().find("position_embeddings") == std::string::npos) {
            continue;
        }
        if (!matcher.match(op->output(0))) {
            continue;
        }
        const auto& table_shape = matcher.get_pattern_value_map().at(table).get_partial_shape();
        if (table_shape.rank().is_static() && table_shape.size() == 2 && table_shape[0].is_static()) {
            return static_cast<uint32_t>(table_shape[0].get_length());
        }
    }
    return std::nullopt;
}
