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
    if (model->has_rt_info({"gguf_mmproj", "audio.projector"})) {
        auto audio = model->clone();
        AdaptMmprojToGenAI(AdaptMmprojToGenAI::Modality::AUDIO).run_on_model(audio);
        adapt_audio_layout(audio);
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
