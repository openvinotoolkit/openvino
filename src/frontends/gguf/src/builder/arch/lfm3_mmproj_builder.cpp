// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm3_mmproj_builder.hpp"

#include <iomanip>
#include <limits>
#include <sstream>

#include "builder/api/metadata_store.hpp"
#include "builder/gguf_graph.hpp"
#include "builder/projector_registry.hpp"
#include "lfm3_audio_generator.hpp"
#include "projector_adapter.hpp"

namespace ov::frontend::gguf {
namespace {
std::string generation_type(const GgufMetadata& metadata) {
    const auto common = metadata.get_str("clip.projector_type").value_or("");
    return common.empty() ? metadata.get_str("clip.gen.audio.projector_type").value_or("") : common;
}

class Lfm3MmprojBuilder : public ModelBuilder {
public:
    explicit Lfm3MmprojBuilder(const BuildContext& context) : ctx(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext g(ctx);
        ModelContract contract;
        contract.batchable_recurrent = false;
        g.configure_model_contract(contract);
        ProjectorRegistry defaults;
        const auto& registry = ctx.projectors ? *ctx.projectors : defaults;
        ModelAdapters adapters;
        std::map<std::string, std::string> config{{"vision.auxiliary_count", "0"}, {"lfm3", "true"}};
        bool any = false;
        for (const auto* modality : {"vision", "audio"}) {
            if (!ctx.metadata.get_bool("clip.has_" + std::string(modality) + "_encoder").value_or(false))
                continue;
            const auto type = resolve_projector_type(ctx.metadata, modality);
            const auto definition = registry.find(ctx.metadata, modality, type);
            OPENVINO_ASSERT(definition, "[GGUF] unsupported LFM3 ", modality, " projector '", type, "'");
            auto result = definition->build(g);
            OPENVINO_ASSERT(result.output, "[GGUF] LFM3 projector returned no output");
            g.set_output(result.output, std::string(modality) + ".embeddings");
            for (const auto& entry : result.config) {
                OPENVINO_ASSERT(entry.first.rfind(std::string(modality) + ".", 0) == 0,
                                "[GGUF] projector metadata must use its modality prefix");
                config[entry.first] = entry.second;
            }
            config[std::string(modality) + ".projector"] = type;
            config[std::string(modality) + ".genai_layout"] = projector_layout_name(
                result.genai_layout.value_or(definition->genai_layout.value_or(ProjectorLayout::Unsupported)));
            if (auto callback = result.genai_adapter.value_or(definition->genai_adapter)) {
                auto adapter = std::make_shared<ModelAdapter>();
                adapter->transform = std::move(callback);
                adapters[std::string(modality) + ".embeddings"] = std::move(adapter);
            }
            any = true;
        }
        if (ctx.metadata.get_bool("clip.has_gen_audio_encoder").value_or(false)) {
            OPENVINO_ASSERT(generation_type(ctx.metadata) == "lfm3a_gen",
                            "[GGUF] unsupported LFM3 audio generation projector");
            build_lfm3_audio_generation(g);
            config["audio_generation.projector"] = "lfm3a_gen";
            any = true;
        }
        OPENVINO_ASSERT(any, "[GGUF] LFM3 mmproj has no encoder");
        auto graph = g.finish();
        graph->model_adapters = std::move(adapters);
        for (const auto& [key, value] : detail::MetadataAccess::get(ctx.metadata).map) {
            if (key.rfind("clip.", 0) != 0)
                continue;
            std::ostringstream stream;
            stream.precision(std::numeric_limits<double>::max_digits10);
            if (auto text = ctx.metadata.get_str(key)) {
                stream << *text;
            } else if (auto number = ctx.metadata.get_int(key)) {
                stream << *number;
            } else if (auto number = ctx.metadata.get_float(key)) {
                stream << *number;
            } else if (std::holds_alternative<std::vector<std::string>>(value)) {
                for (const auto& text : ctx.metadata.get_str_array(key))
                    stream << text.size() << ':' << text;
                graph->mmproj_config[key + ".encoding"] = std::string("length-prefixed-strings");
            } else if (auto array = ctx.metadata.get_int_array(key); !array.empty()) {
                for (size_t i = 0; i < array.size(); ++i)
                    stream << (i ? "," : "") << array[i];
            } else {
                const auto floats = ctx.metadata.get_float_array(key);
                for (size_t i = 0; i < floats.size(); ++i)
                    stream << (i ? "," : "") << floats[i];
            }
            graph->mmproj_config[key] = stream.str();
        }
        for (const auto& entry : config)
            graph->mmproj_config[entry.first] = entry.second;
        return graph;
    }

private:
    BuildContext ctx;
};
}  // namespace

bool is_lfm3_mmproj(const GgufMetadata& metadata) {
    if (metadata.architecture() != "clip")
        return false;
    return resolve_projector_type(metadata, "vision") == "lfm3vl" ||
           resolve_projector_type(metadata, "audio") == "lfm3a" || generation_type(metadata) == "lfm3a_gen";
}

std::shared_ptr<ModelBuilder> make_lfm3_mmproj_builder(const BuildContext& context) {
    return std::make_shared<Lfm3MmprojBuilder>(context);
}
}  // namespace ov::frontend::gguf
