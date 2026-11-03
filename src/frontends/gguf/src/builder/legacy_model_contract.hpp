// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::frontend::gguf {

inline std::shared_ptr<ov::op::v8::Gather> token_embedding_lookup(const std::shared_ptr<ov::Node>& node) {
    const auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(node);
    if (!unsqueeze)
        return nullptr;
    auto source = unsqueeze->input_value(0).get_node_shared_ptr();
    if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source))
        source = convert->input_value(0).get_node_shared_ptr();
    auto gather = ov::as_type_ptr<ov::op::v8::Gather>(source);
    if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(source)) {
        const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(reshape->get_input_node_shared_ptr(0));
        if (multiply)
            gather = ov::as_type_ptr<ov::op::v8::Gather>(multiply->get_input_node_shared_ptr(1));
    }
    if (!gather)
        return nullptr;
    auto indices = gather->input_value(1).get_node_shared_ptr();
    if (const auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(indices))
        indices = squeeze->input_value(0).get_node_shared_ptr();
    const auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(indices);
    return parameter && parameter->output(0).get_names().count("inp_tokens") ? gather : nullptr;
}

struct LegacyEmbeddingBoundaries {
    std::shared_ptr<ov::Node> token;
    std::shared_ptr<ov::Node> per_layer;
    int64_t per_layer_count = 0;
};

inline bool has_legacy_embedding_name(const ov::Node& node, const std::string& name) {
    const auto& friendly_name = node.get_friendly_name();
    const auto suffix = "_" + name;
    return friendly_name == name ||
           (friendly_name.size() > suffix.size() &&
            friendly_name.compare(friendly_name.size() - suffix.size(), suffix.size(), suffix) == 0);
}

inline LegacyEmbeddingBoundaries resolve_legacy_embedding_boundaries(const std::shared_ptr<ov::Model>& model,
                                                                     bool find_token,
                                                                     bool find_per_layer) {
    LegacyEmbeddingBoundaries boundaries;
    const auto architecture = model->has_rt_info({"gguf_architecture"})
                                  ? model->get_rt_info<std::string>({"gguf_architecture"})
                                  : std::string{};
    const auto operations = model->get_ops();
    const auto depends_on = [](const std::shared_ptr<ov::Node>& source, const std::shared_ptr<ov::Node>& target) {
        std::vector<std::shared_ptr<ov::Node>> pending{source};
        std::unordered_set<const ov::Node*> visited;
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            if (current == target)
                return true;
            if (!visited.insert(current.get()).second)
                continue;
            for (const auto& input : current->input_values())
                pending.push_back(input.get_node_shared_ptr());
        }
        return false;
    };
    const auto find_lookup = [&](const std::string& weight_name) {
        std::shared_ptr<ov::Node> found;
        const std::string lookup_name = weight_name == "token_embd.weight" ? "embd" : "pe_tok_flat";
        for (const auto& node : operations) {
            const auto lookup = token_embedding_lookup(node);
            if (!lookup)
                continue;
            auto table = lookup->get_input_node_shared_ptr(0);
            if (!has_legacy_embedding_name(*table, weight_name)) {
                if (const auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(table))
                    table = squeeze->get_input_node_shared_ptr(0);
                if (!has_legacy_embedding_name(*table, weight_name))
                    continue;
            }
            if (weight_name == "token_embd.weight" && has_legacy_embedding_name(*table, "per_layer_token_embd.weight"))
                continue;
            OPENVINO_ASSERT(!found, "[GGUF] ambiguous legacy embedding lookup ", weight_name);
            found = node;
            std::unordered_map<const ov::Node*, std::pair<bool, bool>> dependencies;
            std::function<std::pair<bool, bool>(const std::shared_ptr<ov::Node>&)> token_only;
            token_only = [&](const std::shared_ptr<ov::Node>& current) -> std::pair<bool, bool> {
                if (current == node)
                    return {true, true};
                if (const auto saved = dependencies.find(current.get()); saved != dependencies.end())
                    return saved->second;
                bool valid = ov::is_type<ov::op::v0::Constant>(current) || ov::is_type<ov::op::v1::Add>(current) ||
                             ov::is_type<ov::op::v8::Gather>(current) || ov::is_type<ov::op::v0::MatMul>(current) ||
                             ov::is_type<ov::op::v1::Subtract>(current) || ov::is_type<ov::op::v1::Multiply>(current) ||
                             ov::is_type<ov::op::v0::Concat>(current) || ov::is_type<ov::op::v1::Split>(current) ||
                             ov::is_type<ov::op::v1::Reshape>(current) || ov::is_type<ov::op::v1::Transpose>(current) ||
                             ov::is_type<ov::op::v0::Convert>(current) ||
                             ov::is_type<ov::op::v1::ConvertLike>(current) ||
                             ov::is_type<ov::op::v0::ShapeOf>(current) || ov::is_type<ov::op::v3::ShapeOf>(current);
                bool from_lookup = false;
                for (const auto& input : current->input_values()) {
                    const auto dependency = token_only(input.get_node_shared_ptr());
                    valid = valid && dependency.first;
                    from_lookup = from_lookup || dependency.second;
                }
                return dependencies.emplace(current.get(), std::make_pair(valid, from_lookup)).first->second;
            };
            for (const auto& candidate : operations) {
                if (!has_legacy_embedding_name(*candidate, lookup_name) ||
                    (lookup_name == "embd" && has_legacy_embedding_name(*candidate, "per_layer_embd")))
                    continue;
                const auto dependency = token_only(candidate);
                if (!dependency.second)
                    continue;
                OPENVINO_ASSERT(
                    dependency.first && candidate->get_output_size() == 1 &&
                        candidate->get_output_partial_shape(0).compatible(node->get_output_partial_shape(0)),
                    "[GGUF] unsupported legacy token-only embedding endpoint ",
                    lookup_name);
                if (depends_on(candidate, found)) {
                    found = candidate;
                } else {
                    OPENVINO_ASSERT(depends_on(found, candidate),
                                    "[GGUF] ambiguous legacy complete embedding endpoint ",
                                    lookup_name);
                }
            }
        }
        return found;
    };
    const auto find_scaled = [&](const std::shared_ptr<ov::Node>& lookup, const std::string& name) {
        std::shared_ptr<ov::Node> found;
        for (const auto& node : operations) {
            const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(node);
            if (!multiply || !has_legacy_embedding_name(*multiply, name) ||
                multiply->get_input_node_shared_ptr(0) != lookup)
                continue;
            const auto scale = ov::as_type_ptr<ov::op::v0::Constant>(multiply->get_input_node_shared_ptr(1));
            if (!scale || !scale->get_shape().empty())
                continue;
            OPENVINO_ASSERT(!found, "[GGUF] ambiguous legacy scaled embedding boundary ", name);
            found = node;
        }
        return found;
    };
    if (find_token) {
        boundaries.token = find_lookup("token_embd.weight");
        if (boundaries.token && (architecture == "gemma3" || architecture == "gemma4")) {
            if (const auto scaled = find_scaled(boundaries.token, "embd_scaled"))
                boundaries.token = scaled;
        }
    }
    if (find_per_layer && architecture == "gemma4") {
        const auto lookup = find_lookup("per_layer_token_embd.weight");
        if (!lookup)
            return boundaries;
        const auto scaled = find_scaled(lookup, "pe_tok_flat_scaled");
        if (!scaled)
            return boundaries;
        for (const auto& node : operations) {
            const auto split = ov::as_type_ptr<ov::op::v1::Reshape>(node);
            if (!split || !has_legacy_embedding_name(*split, "pe_tok_split") ||
                split->get_input_node_shared_ptr(0) != scaled || !split->get_special_zero())
                continue;
            const auto target = ov::as_type_ptr<ov::op::v0::Constant>(split->get_input_node_shared_ptr(1));
            if (!target)
                continue;
            const auto dimensions = target->cast_vector<int64_t>();
            const auto& shape = scaled->get_output_partial_shape(0);
            if (dimensions.size() != 4 || dimensions[0] != 0 || dimensions[1] != -1 || dimensions[2] <= 0 ||
                dimensions[3] <= 0 || shape.rank() != 4 || shape[3].is_dynamic() ||
                shape[3].get_length() / dimensions[2] != dimensions[3] || shape[3].get_length() % dimensions[2] != 0)
                continue;
            OPENVINO_ASSERT(!boundaries.per_layer, "[GGUF] ambiguous legacy per-layer embedding split");
            boundaries.per_layer = scaled;
            boundaries.per_layer_count = dimensions[2];
        }
    }
    return boundaries;
}

inline ModelContract resolve_legacy_model_contract(const std::shared_ptr<ov::Model>& model,
                                                   std::optional<int64_t> embedding_width) {
    ModelContract contract;
    if (!model->has_rt_info({"gguf_architecture"}))
        return contract;
    const auto architecture = model->get_rt_info<std::string>({"gguf_architecture"});
    if (architecture == "lfm2" || architecture == "lfm2moe")
        contract.batchable_recurrent = true;
    if (architecture == "gemma3")
        contract.image_attention = ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING;
    if (architecture == "gemma4" && embedding_width && *embedding_width != 1536 && *embedding_width != 2560)
        contract.image_attention = ModelContract::ImageAttentionMode::SLIDING_ONLY;
    return contract;
}

}  // namespace ov::frontend::gguf
