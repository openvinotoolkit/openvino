// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include <cstring>

#include "openvino/op/gather.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

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
    // The position embedding is a Gather(weight[max_position_embeddings, hidden], position_ids).
    // Find that Gather by name and read dim 0 of its data input (port 0), walking through any
    // Convert/decompression nodes in between.
    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::v8::Gather>(op)) {
            continue;
        }
        if (op->get_friendly_name().find("position_embeddings") == std::string::npos) {
            continue;
        }
        auto src = op->input_value(0);  // the position-embedding weight (possibly behind Convert)
        for (int hops = 0; hops < 4 && src.get_node(); ++hops) {
            const auto& ps = src.get_partial_shape();
            if (ps.rank().is_static() && ps.size() == 2 && ps[0].is_static()) {
                return static_cast<uint32_t>(ps[0].get_length());
            }
            if (src.get_node()->get_input_size() == 0) {
                break;
            }
            src = src.get_node()->input_value(0);
        }
    }
    return std::nullopt;
}
