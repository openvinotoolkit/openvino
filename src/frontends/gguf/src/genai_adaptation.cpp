// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_set>

#include "builder/arch/projector_adapter.hpp"
#include "model_adapter_attribute.hpp"
#include "model_contract.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "pass/prune_orphaned_parameters.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {
namespace {

using pass::AdaptMmprojToGenAI;

ModelMap adapt_decoder(const std::shared_ptr<ov::Model>& model, const GenAIAdaptOptions& options) {
    if (model->has_rt_info("gguf_mtp_components")) {
        const auto info = model->get_rt_info<ov::AnyMap>({"gguf_mtp_components"});
        const auto result_count = info.at("target_results").as<int64_t>();
        const auto parameter_count = info.at("target_parameters").as<int64_t>();
        const bool has_state_info =
            info.count("target_sinks") && info.count("target_variables") && info.count("draft_variable_prefix");
        OPENVINO_ASSERT(has_state_info || (!info.count("target_sinks") && !info.count("target_variables") &&
                                           !info.count("draft_variable_prefix") && model->get_sinks().empty() &&
                                           model->get_variables().empty()),
                        "[GGUF] invalid MTP state boundaries");
        const auto sink_count = has_state_info ? info.at("target_sinks").as<int64_t>() : 0;
        const auto variable_count = has_state_info ? info.at("target_variables").as<int64_t>() : 0;
        OPENVINO_ASSERT(result_count > 0 && parameter_count > 0 &&
                            static_cast<size_t>(result_count) < model->get_results().size() &&
                            static_cast<size_t>(parameter_count) < model->get_parameters().size() && sink_count >= 0 &&
                            static_cast<size_t>(sink_count) <= model->get_sinks().size() && variable_count >= 0 &&
                            static_cast<size_t>(variable_count) <= model->get_variables().size(),
                        "[GGUF] invalid MTP component boundaries");
        const auto select = [&](bool target) {
            auto branch = model->clone();
            const auto results = branch->get_results();
            const auto parameters = branch->get_parameters();
            const auto sinks = branch->get_sinks();
            const auto variables = branch->get_variables();
            for (size_t i = 0; i < results.size(); ++i)
                if ((i < static_cast<size_t>(result_count)) != target)
                    branch->remove_result(results[i]);
            for (size_t i = 0; i < sinks.size(); ++i)
                if ((i < static_cast<size_t>(sink_count)) != target)
                    branch->remove_sink(sinks[i]);
            for (size_t i = 0; i < parameters.size(); ++i)
                if ((i < static_cast<size_t>(parameter_count)) != target)
                    branch->remove_parameter(parameters[i]);
            for (size_t i = 0; i < variables.size(); ++i)
                if ((i < static_cast<size_t>(variable_count)) != target)
                    branch->remove_variable(variables[i]);
            if (target) {
                branch->get_rt_info() = model->get_rt_info();
                branch->get_rt_info().erase("gguf_mtp_components");
            } else {
                branch->get_rt_info() = info.at("draft_rt_info").as<ov::AnyMap>();
                if (info.count("draft_name"))
                    branch->set_friendly_name(info.at("draft_name").as<std::string>());
                if (!branch->get_variables().empty())
                    namespace_model_variables(branch, info.at("draft_variable_prefix").as<std::string>(), true);
                const auto io_prefix =
                    info.count("draft_io_prefix") ? info.at("draft_io_prefix").as<std::string>() : std::string{"mtp."};
                namespace_model_boundaries(branch, io_prefix, true);
            }
            branch->validate_nodes_and_infer_types();
            return branch;
        };
        auto models = adapt_decoder(select(true), options);
        auto draft = adapt_decoder(select(false), options);
        models.emplace("mtp_draft", draft.at("language"));
        return models;
    }
    if (!find_parameter(model, "beam_idx")) {
        auto live = std::make_shared<std::unordered_set<const ov::Node*>>();
        pass::SnapshotLiveParameters(live).run_on_model(model);
        pass::GGUFMakeStateful().run_on_model(model);
        // The stateful caches no longer read the lowered write indices.
        pass::PruneParametersOrphanedSince(live).run_on_model(model);
    }
    pass::AdaptToGenAI adapter(options.input);
    OPENVINO_ASSERT(adapter.run_on_model(model),
                    "[GGUF] adapt_to_genai requires a model converted by the GGUF frontend; the model is already "
                    "adapted or has no GGUF decoder inputs");
    ModelMap models{{"language", model}};
    if (const auto& embeddings = adapter.get_embedding_model())
        models.emplace("text_embeddings", embeddings);
    if (const auto& per_layer = adapter.get_per_layer_embedding_model())
        models.emplace("text_embeddings_per_layer", per_layer);
    return models;
}

bool has_genai_vision_layout(const std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->input_value(0).get_names().count("vision.embeddings") && get_model_adapter(result))
            return true;
    }
    if (model->has_rt_info({"gguf_mmproj", "vision.genai_layout"}))
        return model->get_rt_info<std::string>({"gguf_mmproj", "vision.genai_layout"}) !=
               projector_layout_name(ProjectorLayout::Unsupported);
    const auto projector = model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"});
    return builtin_projector_layout(projector) != ProjectorLayout::Unsupported;
}

bool has_genai_adapter(const std::shared_ptr<ov::Model>& model, const std::string& modality) {
    const auto embedding_name = modality + ".embeddings";
    for (const auto& result : model->get_results()) {
        if (result->input_value(0).get_names().count(embedding_name) && get_model_adapter(result))
            return true;
    }
    return false;
}

bool has_mmproj_branch(const std::shared_ptr<ov::Model>& model, const std::string& modality) {
    const auto supported = modality + ".supported";
    return model->has_rt_info({"gguf_mmproj", modality + ".projector"}) &&
           (!model->has_rt_info({"gguf_mmproj", supported}) ||
            model->get_rt_info<std::string>({"gguf_mmproj", supported}) != "false");
}

ModelMap adapt_mmproj(const std::shared_ptr<ov::Model>& model) {
    ModelMap models;
    const auto add = [&](const std::string& name, const std::shared_ptr<ov::Model>& adapted) {
        OPENVINO_ASSERT(models.emplace(name, adapted).second, "[GGUF] mmproj declares model '", name, "' twice");
    };
    if (has_mmproj_branch(model, "vision")) {
        auto vision = model->clone();
        if (has_genai_vision_layout(model)) {
            AdaptMmprojToGenAI adapter(AdaptMmprojToGenAI::Modality::VISION,
                                       AdaptMmprojToGenAI::Layout::VISION_ENCODERS);
            adapter.run_on_model(vision);
            for (const auto& [name, encoder] : adapter.get_vision_models())
                add(name, encoder);
        } else {
            AdaptMmprojToGenAI(AdaptMmprojToGenAI::Modality::VISION).run_on_model(vision);
            add("vision_embeddings", vision);
        }
    }
    if (has_mmproj_branch(model, "audio")) {
        auto audio = model->clone();
        const bool has_adapter = has_genai_adapter(audio, "audio");
        AdaptMmprojToGenAI adapter(AdaptMmprojToGenAI::Modality::AUDIO);
        adapter.run_on_model(audio);
        if (!has_adapter)
            adapt_audio_layout(audio);
        const auto& audio_models = adapter.get_vision_models();
        for (const auto& [name, component] : audio_models)
            add(name, component);
        if (audio_models.empty())
            add("audio_embeddings", audio);
    }
    if (read_model_contract(model).component_groups.count("audio_generation")) {
        AdaptMmprojToGenAI adapter(AdaptMmprojToGenAI::Modality::AUDIO_GENERATION);
        adapter.run_on_model(model->clone());
        for (const auto& [name, component] : adapter.get_audio_generation_models())
            add("audio_generation_" + name, component);
    }
    OPENVINO_ASSERT(!models.empty(), "[GGUF] mmproj has no encoder for GenAI");
    return models;
}

}  // namespace

ModelMap adapt_to_genai(const std::shared_ptr<ov::Model>& model, const GenAIAdaptOptions& options) {
    OPENVINO_ASSERT(model, "[GGUF] adapt_to_genai requires a model");
    return model->has_rt_info("gguf_mmproj") ? adapt_mmproj(model) : adapt_decoder(model, options);
}

}  // namespace ov::frontend::gguf
