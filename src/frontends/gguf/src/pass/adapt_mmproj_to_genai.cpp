// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "builder/arch/projector_adapter.hpp"
#include "model_adapter_attribute.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "pass/prune_orphaned_parameters.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::pass {
namespace {
using Models = std::map<std::string, std::shared_ptr<ov::Model>>;

bool is_lfm3(const std::shared_ptr<ov::Model>& model) {
    return model->has_rt_info({"gguf_mmproj", "lfm3"}) &&
           model->get_rt_info<std::string>({"gguf_mmproj", "lfm3"}) == "true";
}

ov::ResultVector retain_lfm3_states(const std::shared_ptr<ov::Model>& model, const std::string& prefix) {
    ov::ResultVector retained;
    auto& info = model->get_rt_info();
    const auto key = gguf_recurrent_states_key();
    if (const auto it = info.find(key); it != info.end()) {
        const auto pairs = it->second.as<std::vector<std::string>>();
        std::vector<std::string> selected;
        for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
            if (prefix.empty() || pairs[i].rfind(prefix, 0) != 0)
                continue;
            selected.insert(selected.end(), {pairs[i], pairs[i + 1]});
            for (const auto& result : model->get_results()) {
                if (result->get_friendly_name() == pairs[i + 1] ||
                    result->input_value(0).get_names().count(pairs[i + 1]))
                    retained.push_back(result);
            }
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

Models split_lfm3_generation(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(is_lfm3(model), "[GGUF] AUDIO_GENERATION requires an LFM3 mmproj");
    Models models;
    for (const auto& [name, output] : std::map<std::string, std::string>{{"depthformer", "audio.depth_logits"},
                                                                         {"feedback", "audio.feedback"},
                                                                         {"codec", "audio.spectrogram"}}) {
        auto branch = model->clone();
        auto live = std::make_shared<std::unordered_set<const ov::Node*>>();
        SnapshotLiveParameters(live).run_on_model(branch);
        const auto retained = retain_lfm3_states(branch, name == "codec" ? "audio.codec.state." : "");
        auto results = branch->get_results();
        bool found = false;
        for (const auto& result : results) {
            const auto names = result->input_value(0).get_names();
            if (names.count(output)) {
                found = true;
                continue;
            }
            if (name == "codec" && names.count("audio.istft_window"))
                continue;
            if (std::find(retained.begin(), retained.end(), result) == retained.end())
                branch->remove_result(result);
        }
        OPENVINO_ASSERT(found, "[GGUF] mmproj has no ", output, " output");
        PruneParametersOrphanedSince(live).run_on_model(branch);
        branch->set_friendly_name("lfm3_" + name);
        branch->validate_nodes_and_infer_types();
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
    model->add_sinks(adapted->get_sinks());
    model->add_variables(adapted->get_variables());
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
    const auto retained_states = is_lfm3(model)
                                     ? retain_lfm3_states(model, modality == Modality::AUDIO ? "audio.state." : "")
                                     : ov::ResultVector{};
    const auto results = model->get_results();
    for (const auto& result : results)
        model->remove_result(result);
    model->add_results(retained_states);
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
    PruneParametersOrphanedSince(live_before).run_on_model(model);
    for (const auto& p : model->get_parameters()) {
        const auto& name = p->get_friendly_name();
        if (name.rfind(prefix, 0) == 0 && !(is_lfm3(model) && name.rfind("audio.state.", 0) == 0))
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
        auto models = split_lfm3_generation(model);
        const auto primary = models.at("depthformer");
        replace_model(model, primary);
        models["depthformer"] = model;
        m_audio_generation_models = std::move(models);
        return true;
    }
    if (m_layout == Layout::EMBEDDINGS && !get_model_adapter(embedding_result(model, m_modality))) {
        adapt_mmproj(model, m_modality, m_layout);
        remove_model_adapters(model);
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
    model->add_sinks(adapted->get_sinks());
    model->add_variables(adapted->get_variables());
    model->get_rt_info() = adapted->get_rt_info();
    for (auto& [name, encoder] : vision_models) {
        if (encoder == adapted)
            encoder = model;
    }
    m_vision_models = std::move(vision_models);
    return true;
}
}  // namespace ov::frontend::gguf::pass
