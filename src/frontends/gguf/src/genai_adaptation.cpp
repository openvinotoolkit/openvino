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
        OPENVINO_ASSERT(result_count > 0 && parameter_count > 0 &&
                            static_cast<size_t>(result_count) < model->get_results().size() &&
                            static_cast<size_t>(parameter_count) < model->get_parameters().size(),
                        "[GGUF] invalid MTP component boundaries");
        const auto select = [&](bool target) {
            auto branch = model->clone();
            const auto results = branch->get_results();
            const auto parameters = branch->get_parameters();
            for (size_t i = 0; i < results.size(); ++i)
                if ((i < static_cast<size_t>(result_count)) != target)
                    branch->remove_result(results[i]);
            for (size_t i = 0; i < parameters.size(); ++i)
                if ((i < static_cast<size_t>(parameter_count)) != target)
                    branch->remove_parameter(parameters[i]);
            if (target) {
                branch->get_rt_info() = model->get_rt_info();
                branch->get_rt_info().erase("gguf_mtp_components");
            } else {
                branch->get_rt_info() = info.at("draft_rt_info").as<ov::AnyMap>();
                const auto unprefix = [](const std::string& name) {
                    OPENVINO_ASSERT(name.rfind("mtp.", 0) == 0, "[GGUF] invalid MTP boundary name: ", name);
                    return name.substr(4);
                };
                const auto restore_names = [&](const ov::Output<ov::Node>& output) {
                    std::unordered_set<std::string> names;
                    for (const auto& name : output.get_names())
                        names.insert(unprefix(name));
                    output.get_tensor().set_names(names);
                };
                for (const auto& parameter : branch->get_parameters()) {
                    parameter->set_friendly_name(unprefix(parameter->get_friendly_name()));
                    restore_names(parameter->output(0));
                }
                for (const auto& result : branch->get_results()) {
                    result->set_friendly_name(unprefix(result->get_friendly_name()));
                    restore_names(result->input_value(0));
                }
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

ModelMap adapt_mmproj(const std::shared_ptr<ov::Model>& model) {
    ModelMap models;
    const auto add = [&](const std::string& name, const std::shared_ptr<ov::Model>& adapted) {
        OPENVINO_ASSERT(models.emplace(name, adapted).second, "[GGUF] mmproj declares model '", name, "' twice");
    };
    if (model->has_rt_info({"gguf_mmproj", "vision.projector"})) {
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
    if (model->has_rt_info({"gguf_mmproj", "audio.projector"}) &&
        (!model->has_rt_info({"gguf_mmproj", "audio.supported"}) ||
         model->get_rt_info<std::string>({"gguf_mmproj", "audio.supported"}) != "false")) {
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
