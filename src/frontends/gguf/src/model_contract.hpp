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

inline int64_t model_contract_integer(const std::string& value, int64_t maximum) {
    OPENVINO_ASSERT(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "[GGUF] invalid model contract integer");
    int64_t number = 0;
    for (const auto character : value) {
        const auto digit = character - '0';
        OPENVINO_ASSERT(number <= maximum / 10 && (number != maximum / 10 || digit <= maximum % 10),
                        "[GGUF] model contract integer out of range");
        number = number * 10 + digit;
    }
    return number;
}

inline void validate_genai_contract(const ModelContract::GenAIContract& contract) {
    OPENVINO_ASSERT(contract.position_layout == ModelContract::GenAIContract::PositionLayout::THREE_AXIS ||
                        contract.position_layout == ModelContract::GenAIContract::PositionLayout::SEQUENCE_THREE_AXIS,
                    "[GGUF] invalid GenAI position layout");
    OPENVINO_ASSERT(contract.auxiliary_outputs.size() <= 1024 && contract.preserved_inputs.size() <= 1024,
                    "[GGUF] too many GenAI contract ports");
    std::unordered_set<std::string> sources, names, inputs;
    for (const auto& output : contract.auxiliary_outputs) {
        OPENVINO_ASSERT(!output.source_tensor.empty() && !output.name.empty() && output.name != "logits" &&
                            sources.insert(output.source_tensor).second && names.insert(output.name).second,
                        "[GGUF] invalid or duplicate GenAI auxiliary output");
        OPENVINO_ASSERT(!output.squeeze_axis || (*output.squeeze_axis >= 0 && *output.squeeze_axis <= 3),
                        "[GGUF] invalid GenAI output squeeze axis");
    }
    for (const auto& input : contract.preserved_inputs)
        OPENVINO_ASSERT(!input.empty() && inputs.insert(input).second,
                        "[GGUF] invalid or duplicate preserved GenAI input");
}

inline void validate_model_contract(const ModelContract& contract) {
    if (contract.genai)
        validate_genai_contract(*contract.genai);
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
    const bool supports_genai =
        contract.supports_genai_adaptation
            ? *contract.supports_genai_adaptation
            : (contract.genai ? contract.genai->enabled : resolve_legacy_genai_adaptation(model));
    model->get_rt_info()["gguf_model_contract"] =
        ov::AnyMap{{"batchable_recurrent", recurrent},
                   {"image_attention", image_attention},
                   {"sequence_outputs", contract.sequence_outputs},
                   {"component_groups", component_groups},
                   {"supports_genai_adaptation", std::string(supports_genai ? "true" : "false")}};
    if (contract.genai) {
        const auto& genai = *contract.genai;
        ov::AnyMap outputs, inputs;
        for (size_t i = 0; i < genai.auxiliary_outputs.size(); ++i) {
            const auto& output = genai.auxiliary_outputs[i];
            outputs["port_" + std::to_string(i)] =
                ov::AnyMap{{"source_tensor", output.source_tensor},
                           {"name", output.name},
                           {"squeeze_axis", output.squeeze_axis ? std::to_string(*output.squeeze_axis) : "none"}};
        }
        for (size_t i = 0; i < genai.preserved_inputs.size(); ++i)
            inputs["port_" + std::to_string(i)] = genai.preserved_inputs[i];
        model->set_rt_info(
            ov::AnyMap{{"version", std::string{"1"}},
                       {"enabled", std::string{supports_genai ? "true" : "false"}},
                       {"position_layout",
                        std::string{genai.position_layout == ModelContract::GenAIContract::PositionLayout::THREE_AXIS
                                        ? "three_axis"
                                        : "sequence_three_axis"}},
                       {"causal_attention", std::string{genai.causal_attention ? "true" : "false"}},
                       {"attention_context_input", genai.attention_context_input},
                       {"logits_first", std::string{genai.logits_first ? "true" : "false"}},
                       {"output_count", std::to_string(genai.auxiliary_outputs.size())},
                       {"input_count", std::to_string(genai.preserved_inputs.size())},
                       {"outputs", outputs},
                       {"inputs", inputs}},
            {"gguf_model_contract", "genai"});
    }
}

inline ModelContract read_model_contract(const std::shared_ptr<ov::Model>& model,
                                         std::optional<int64_t> embedding_width = std::nullopt) {
    if (!model->has_rt_info({"gguf_model_contract"})) {
        auto contract = resolve_legacy_model_contract(model, embedding_width);
        validate_model_contract(contract);
        return contract;
    }
    ModelContract contract;
    const bool has_genai = model->has_rt_info({"gguf_model_contract", "genai"});
    if (has_genai) {
        const auto value = [&](const std::string& name) {
            return model->get_rt_info<std::string>({"gguf_model_contract", "genai", name});
        };
        const auto boolean = [&](const std::string& name) {
            const auto text = value(name);
            OPENVINO_ASSERT(text == "true" || text == "false", "[GGUF] invalid GenAI contract boolean ", name);
            return text == "true";
        };
        OPENVINO_ASSERT(value("version") == "1", "[GGUF] unsupported GenAI model contract version");
        ModelContract::GenAIContract genai;
        genai.enabled = boolean("enabled");
        genai.causal_attention = boolean("causal_attention");
        genai.logits_first = boolean("logits_first");
        genai.attention_context_input = value("attention_context_input");
        const auto layout = value("position_layout");
        OPENVINO_ASSERT(layout == "three_axis" || layout == "sequence_three_axis",
                        "[GGUF] invalid GenAI position layout");
        if (layout == "three_axis")
            genai.position_layout = ModelContract::GenAIContract::PositionLayout::THREE_AXIS;
        const auto output_count = model_contract_integer(value("output_count"), 1024);
        for (int64_t i = 0; i < output_count; ++i) {
            const auto field = [&](const std::string& name) {
                return model->get_rt_info<std::string>(
                    {"gguf_model_contract", "genai", "outputs", "port_" + std::to_string(i), name});
            };
            const auto axis = field("squeeze_axis");
            genai.auxiliary_outputs.push_back(
                {field("source_tensor"),
                 field("name"),
                 axis == "none" ? std::nullopt : std::optional<int64_t>{model_contract_integer(axis, 3)}});
        }
        const auto input_count = model_contract_integer(value("input_count"), 1024);
        for (int64_t i = 0; i < input_count; ++i)
            genai.preserved_inputs.push_back(model->get_rt_info<std::string>(
                {"gguf_model_contract", "genai", "inputs", "port_" + std::to_string(i)}));
        contract.genai = std::move(genai);
    } else {
        contract.genai = resolve_legacy_genai_contract(model);
    }
    if (model->has_rt_info({"gguf_model_contract", "supports_genai_adaptation"})) {
        const auto value = model->get_rt_info<std::string>({"gguf_model_contract", "supports_genai_adaptation"});
        OPENVINO_ASSERT(value == "true" || value == "false", "[GGUF] invalid GenAI adaptation contract");
        contract.supports_genai_adaptation = value == "true";
    } else {
        contract.supports_genai_adaptation =
            has_genai ? contract.genai->enabled : resolve_legacy_genai_adaptation(model);
    }
    contract.genai->enabled = *contract.supports_genai_adaptation;
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
