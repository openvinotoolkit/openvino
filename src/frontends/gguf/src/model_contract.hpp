// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>
#include <utility>

#include "builder/legacy_model_contract.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/decoder.hpp"

namespace ov::frontend::gguf {

inline void validate_model_contract(const ModelContract& contract) {
    const auto validate_outputs = [](const std::vector<std::string>& outputs) {
        std::unordered_set<std::string> names;
        for (const auto& name : outputs)
            OPENVINO_ASSERT(!name.empty() && names.insert(name).second,
                            "[GGUF] model contract output names must be nonempty and unique");
    };
    validate_outputs(contract.sequence_outputs);
    for (const auto& [name, group] : contract.component_groups) {
        OPENVINO_ASSERT(!name.empty(), "[GGUF] model contract component group name must be nonempty");
        OPENVINO_ASSERT(group.models.count(group.primary),
                        "[GGUF] model contract component group primary must name a component");
        for (const auto& [component_name, component] : group.models) {
            OPENVINO_ASSERT(!component_name.empty(), "[GGUF] model contract component name must be nonempty");
            OPENVINO_ASSERT(!component.outputs.empty(), "[GGUF] model contract component outputs must be nonempty");
            validate_outputs(component.outputs);
        }
    }
}

inline void write_model_contract(const std::shared_ptr<ov::Model>& model, const ModelContract& contract) {
    validate_model_contract(contract);
    std::string image_attention;
    switch (contract.image_attention) {
    case ModelContract::ImageAttentionMode::CAUSAL:
        image_attention = "causal";
        break;
    case ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING:
        image_attention = "global_and_sliding";
        break;
    case ModelContract::ImageAttentionMode::SLIDING_ONLY:
        image_attention = "sliding_only";
        break;
    default:
        OPENVINO_THROW("[GGUF] invalid image attention contract");
    }
    const std::string recurrent = !contract.batchable_recurrent   ? "auto"
                                  : *contract.batchable_recurrent ? "true"
                                                                  : "false";
    ov::AnyMap component_groups;
    for (const auto& [name, group] : contract.component_groups) {
        ov::AnyMap models;
        for (const auto& [component_name, component] : group.models)
            models.emplace(component_name,
                           ov::AnyMap{{"outputs", component.outputs}, {"state_prefix", component.state_prefix}});
        component_groups.emplace(name, ov::AnyMap{{"primary", group.primary}, {"models", models}});
    }
    const bool supports_genai = contract.supports_genai_adaptation ? *contract.supports_genai_adaptation
                                                                   : resolve_legacy_genai_adaptation(model);
    model->get_rt_info()["gguf_model_contract"] =
        ov::AnyMap{{"batchable_recurrent", recurrent},
                   {"image_attention", image_attention},
                   {"sequence_outputs", contract.sequence_outputs},
                   {"component_groups", component_groups},
                   {"supports_genai_adaptation", std::string(supports_genai ? "true" : "false")}};
}

inline ModelContract read_model_contract(const std::shared_ptr<ov::Model>& model,
                                         std::optional<int64_t> embedding_width = std::nullopt) {
    if (!model->has_rt_info({"gguf_model_contract"})) {
        auto contract = resolve_legacy_model_contract(model, embedding_width);
        validate_model_contract(contract);
        return contract;
    }
    ModelContract contract;
    if (model->has_rt_info({"gguf_model_contract", "supports_genai_adaptation"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "supports_genai_adaptation"});
        OPENVINO_ASSERT(value == "true" || value == "false", "[GGUF] invalid GenAI adaptation contract");
        contract.supports_genai_adaptation = value == "true";
    } else {
        contract.supports_genai_adaptation = resolve_legacy_genai_adaptation(model);
    }
    if (model->has_rt_info({"gguf_model_contract", "batchable_recurrent"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "batchable_recurrent"});
        OPENVINO_ASSERT(value == "true" || value == "false" || value == "auto",
                        "[GGUF] invalid recurrent batching contract");
        if (value != "auto")
            contract.batchable_recurrent = value == "true";
    }
    if (model->has_rt_info({"gguf_model_contract", "image_attention"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "image_attention"});
        if (value == "global_and_sliding") {
            contract.image_attention = ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING;
        } else if (value == "sliding_only") {
            contract.image_attention = ModelContract::ImageAttentionMode::SLIDING_ONLY;
        } else {
            OPENVINO_ASSERT(value == "causal", "[GGUF] invalid image attention contract");
        }
    }
    if (model->has_rt_info({"gguf_model_contract", "sequence_outputs"}))
        contract.sequence_outputs =
            model->get_rt_info<std::vector<std::string>>({"gguf_model_contract", "sequence_outputs"});
    if (model->has_rt_info({"gguf_model_contract", "component_groups"})) {
        const auto groups = model->get_rt_info<ov::AnyMap>({"gguf_model_contract", "component_groups"});
        for (const auto& [name, value] : groups) {
            const auto& group = value.as<ov::AnyMap>();
            OPENVINO_ASSERT(group.count("primary") && group.count("models"),
                            "[GGUF] model contract component group requires primary and models");
            ModelComponentGroup components;
            components.primary = group.at("primary").as<std::string>();
            for (const auto& [component_name, component_value] : group.at("models").as<ov::AnyMap>()) {
                const auto& component = component_value.as<ov::AnyMap>();
                OPENVINO_ASSERT(component.count("outputs") && component.count("state_prefix"),
                                "[GGUF] model contract component requires outputs and state_prefix");
                components.models.emplace(component_name,
                                          ModelComponentContract{component.at("outputs").as<std::vector<std::string>>(),
                                                                 component.at("state_prefix").as<std::string>()});
            }
            contract.component_groups.emplace(name, std::move(components));
        }
    }
    validate_model_contract(contract);
    return contract;
}

}  // namespace ov::frontend::gguf
