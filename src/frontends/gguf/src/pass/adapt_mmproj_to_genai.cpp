// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "builder/arch/projector_adapter.hpp"
#include "model_adapter_attribute.hpp"
#include "model_contract.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "pass/prune_orphaned_parameters.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::pass {
namespace {
using Models = std::map<std::string, std::shared_ptr<ov::Model>>;

struct ComponentStateSelection {
    ov::ResultVector results;
    std::unordered_set<std::string> parameters;
};

ComponentStateSelection retain_component_states(const std::shared_ptr<ov::Model>& model, const std::string& prefix) {
    ComponentStateSelection retained;
    auto& info = model->get_rt_info();
    const auto key = gguf_recurrent_states_key();
    if (const auto it = info.find(key); it != info.end()) {
        const auto pairs = it->second.as<std::vector<std::string>>();
        OPENVINO_ASSERT(pairs.size() % 2 == 0, "[GGUF] invalid component state declarations");
        std::vector<std::string> selected;
        for (size_t i = 0; i < pairs.size(); i += 2) {
            if (prefix.empty() || pairs[i].rfind(prefix, 0) != 0)
                continue;
            selected.insert(selected.end(), {pairs[i], pairs[i + 1]});
            retained.parameters.insert(pairs[i]);
            bool found = false;
            for (const auto& result : model->get_results()) {
                if (result->get_friendly_name() == pairs[i + 1] ||
                    result->input_value(0).get_names().count(pairs[i + 1])) {
                    retained.results.push_back(result);
                    found = true;
                }
            }
            OPENVINO_ASSERT(found || !find_parameter(model, pairs[i]),
                            "[GGUF] missing component state output ",
                            pairs[i + 1]);
        }
        info[key] = std::move(selected);
    }
    const auto sinks = model->get_sinks();
    for (const auto& sink : sinks) {
        const auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink);
        if (assign && (prefix.empty() || assign->get_variable_id().rfind(prefix, 0) != 0))
            model->remove_sink(sink);
    }
    const auto variables = model->get_variables();
    for (const auto& variable : variables) {
        if (prefix.empty() || variable->get_info().variable_id.rfind(prefix, 0) != 0)
            model->remove_variable(variable);
    }
    return retained;
}

ComponentStateSelection select_component(const std::shared_ptr<ov::Model>& model,
                                         const ModelComponentContract& contract) {
    auto live = std::make_shared<std::unordered_set<const ov::Node*>>();
    SnapshotLiveParameters(live).run_on_model(model);
    auto retained = retain_component_states(model, contract.state_prefix);
    std::unordered_set<std::string> found;
    const auto results = model->get_results();
    for (const auto& result : results) {
        bool selected = false;
        for (const auto& name : contract.outputs) {
            if (result->input_value(0).get_names().count(name) || result->get_friendly_name() == name) {
                selected = true;
                found.insert(name);
                if (!result->output(0).get_names().count(name))
                    result->output(0).add_names({name});
            }
        }
        if (!selected && std::find(retained.results.begin(), retained.results.end(), result) == retained.results.end())
            model->remove_result(result);
    }
    for (const auto& name : contract.outputs)
        OPENVINO_ASSERT(found.count(name), "[GGUF] missing declared component output ", name);
    PruneParametersOrphanedSince(live).run_on_model(model);
    model->validate_nodes_and_infer_types();
    return retained;
}

Models split_components(const std::shared_ptr<ov::Model>& model, const ModelComponentGroup& group) {
    Models models;
    for (const auto& [name, contract] : group.models) {
        auto branch = model->clone();
        select_component(branch, contract);
        branch->set_friendly_name(name);
        remove_model_adapters(branch);
        models.emplace(name, std::move(branch));
    }
    return models;
}

void replace_model(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<ov::Model>& adapted) {
    const auto results = model->get_results();
    const auto parameters = model->get_parameters();
    const auto sinks = model->get_sinks();
    const auto variables = model->get_variables();
    for (const auto& result : results)
        model->remove_result(result);
    for (const auto& parameter : parameters)
        model->remove_parameter(parameter);
    for (const auto& sink : sinks)
        model->remove_sink(sink);
    for (const auto& variable : variables)
        model->remove_variable(variable);
    model->add_parameters(adapted->get_parameters());
    model->add_results(adapted->get_results());
    model->add_variables(adapted->get_variables());
    model->add_sinks(adapted->get_sinks());
    model->get_rt_info() = adapted->get_rt_info();
}

std::shared_ptr<ov::op::v0::Result> embedding_result(const std::shared_ptr<ov::Model>& model,
                                                     AdaptMmprojToGenAI::Modality modality) {
    const std::string prefix = modality == AdaptMmprojToGenAI::Modality::VISION ? "vision." : "audio.";
    std::shared_ptr<ov::op::v0::Result> selected;
    for (const auto& result : model->get_results()) {
        if (result->input_value(0).get_names().count(prefix + "embeddings"))
            selected = result;
    }
    OPENVINO_ASSERT(selected, "[GGUF] mmproj has no ", prefix, "embeddings output");
    return selected;
}

ModelAdapterResult adapt_mmproj(const std::shared_ptr<ov::Model>& model,
                                AdaptMmprojToGenAI::Modality modality,
                                AdaptMmprojToGenAI::Layout layout) {
    using Modality = AdaptMmprojToGenAI::Modality;
    using Layout = AdaptMmprojToGenAI::Layout;
    Models vision_models;
    if (layout == Layout::VISION_ENCODERS)
        OPENVINO_ASSERT(modality == Modality::VISION, "[GGUF] VISION_ENCODERS layout requires vision modality");
    const std::string prefix = modality == Modality::VISION ? "vision." : "audio.";
    const auto selected = embedding_result(model, modality);
    const auto adapter = get_model_adapter(selected);
    if (layout == Layout::VISION_ENCODERS && !adapter)
        require_projector_layout(model);
    auto live_before = std::make_shared<std::unordered_set<const ov::Node*>>();
    SnapshotLiveParameters(live_before).run_on_model(model);
    ComponentStateSelection states;
    ov::ResultVector retained_results;
    const auto contract = read_model_contract(model);
    if (const auto group = contract.component_groups.find(modality == Modality::VISION ? "vision" : "audio");
        group != contract.component_groups.end()) {
        states = select_component(model, group->second.models.at(group->second.primary));
        embedding_result(model, modality);
        for (const auto& result : model->get_results()) {
            if (result != selected) {
                const auto value = result->input_value(0);
                if (ov::as_type_ptr<ov::op::v0::Parameter>(value.get_node_shared_ptr())) {
                    // Separate passthrough output names from renamed input names during cloning.
                    auto passthrough = std::make_shared<ov::op::v0::Convert>(value, value.get_element_type());
                    passthrough->output(0).set_names(result->output(0).get_names());
                    result->set_argument(0, passthrough);
                    result->validate_and_infer_types();
                }
                retained_results.push_back(result);
            }
        }
    }
    const auto results = model->get_results();
    for (const auto& result : results)
        model->remove_result(result);
    auto embeddings = std::make_shared<ov::op::v0::Squeeze>(selected->input_value(0),
                                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    size_t auxiliary_count = 0;
    if (modality == Modality::VISION && model->has_rt_info({"gguf_mmproj", "vision.auxiliary_count"}))
        auxiliary_count = size_t(mmproj_int_metadata(model, "vision.auxiliary_count", 0));
    if (auxiliary_count) {
        auto split = std::make_shared<ov::op::v1::Split>(embeddings,
                                                         ov::op::v0::Constant::create(ov::element::i64, {}, {2}),
                                                         auxiliary_count + 1);
        for (size_t i = 0; i <= auxiliary_count; ++i) {
            split->output(i).set_names({i == 0 ? "image_features" : "deepstack_features." + std::to_string(i - 1)});
            model->add_results({std::make_shared<ov::op::v0::Result>(split->output(i))});
        }
    } else {
        // Only the tensor that becomes a Result is named; the Split path names output 0 instead.
        embeddings->output(0).set_names({modality == Modality::VISION ? "image_features" : "audio_features"});
        model->add_results({std::make_shared<ov::op::v0::Result>(embeddings)});
    }
    model->add_results(retained_results);
    PruneParametersOrphanedSince(live_before).run_on_model(model);
    for (const auto& p : model->get_parameters()) {
        const auto& name = p->get_friendly_name();
        if (name.rfind(prefix, 0) == 0 && !states.parameters.count(name))
            name_output(p, name.substr(prefix.size()));
    }
    model->validate_nodes_and_infer_types();
    if (adapter) {
        OPENVINO_ASSERT(adapter->transform, "[GGUF] invalid projector adapter callback");
        return adapter->transform(model);
    }
    if (layout == Layout::VISION_ENCODERS)
        vision_models = adapt_projector_layout(model);
    return {model, std::move(vision_models)};
}
}  // namespace

bool AdaptMmprojToGenAI::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_vision_models.clear();
    m_audio_generation_models.clear();
    if (m_modality == Modality::AUDIO_GENERATION) {
        OPENVINO_ASSERT(m_layout == Layout::EMBEDDINGS, "[GGUF] AUDIO_GENERATION requires EMBEDDINGS layout");
        const auto contract = read_model_contract(model);
        const auto group = contract.component_groups.find("audio_generation");
        OPENVINO_ASSERT(group != contract.component_groups.end(),
                        "[GGUF] model declares no audio-generation component group");
        auto models = split_components(model, group->second);
        const auto primary = models.at(group->second.primary);
        replace_model(model, primary);
        models[group->second.primary] = model;
        m_audio_generation_models = std::move(models);
        return true;
    }
    // Callbacks and layout construction may fail after rewiring the independent clone.
    auto adaptation = adapt_mmproj(model->clone(), m_modality, m_layout);
    auto adapted = std::move(adaptation.model);
    auto vision_models = std::move(adaptation.models);
    OPENVINO_ASSERT(adapted && (m_layout == Layout::EMBEDDINGS || !vision_models.empty()),
                    "[GGUF] projector adapter returned no models");
    bool primary = vision_models.empty();
    for (const auto& [name, encoder] : vision_models) {
        OPENVINO_ASSERT(!name.empty() && encoder, "[GGUF] projector adapter returned an invalid model");
        encoder->validate_nodes_and_infer_types();
        primary |= encoder == adapted;
    }
    OPENVINO_ASSERT(primary, "[GGUF] projector adapter omitted its primary model");
    adapted->validate_nodes_and_infer_types();
    remove_model_adapters(adapted);
    for (const auto& [name, encoder] : vision_models)
        remove_model_adapters(encoder);
    replace_model(model, adapted);
    for (auto& [name, encoder] : vision_models) {
        if (encoder == adapted)
            encoder = model;
    }
    m_vision_models = std::move(vision_models);
    return true;
}
}  // namespace ov::frontend::gguf::pass
