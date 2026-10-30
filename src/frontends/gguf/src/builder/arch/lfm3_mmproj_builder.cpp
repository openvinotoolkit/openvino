// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm3_mmproj_builder.hpp"

#include "builder/projector_registry.hpp"
#include "lfm3_audio_generator.hpp"
#include "mmproj_builder.hpp"

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
        ModelAdapters adapters;
        std::map<std::string, std::string> config;
        for (const auto& modality : build_mmproj_encoders(g, ctx, adapters, config))
            contract.component_groups[modality] = {
                "encoder",
                {{"encoder", {{modality + ".embeddings"}, modality == "audio" ? "audio.state." : ""}}}};
        if (ctx.metadata.get_bool("clip.has_gen_audio_encoder").value_or(false)) {
            OPENVINO_ASSERT(generation_type(ctx.metadata) == "lfm3a_gen",
                            "[GGUF] unsupported LFM3 audio generation projector");
            build_lfm3_audio_generation(g);
            contract.component_groups["audio_generation"] = {
                "depthformer",
                {{"depthformer", {{"audio.depth_logits"}, ""}},
                 {"feedback", {{"audio.feedback"}, ""}},
                 {"codec", {{"audio.spectrogram", "audio.istft_window"}, "audio.codec.state."}}}};
            config["audio_generation.projector"] = "lfm3a_gen";
        }
        OPENVINO_ASSERT(!contract.component_groups.empty(), "[GGUF] LFM3 mmproj has no encoder");
        g.configure_model_contract(contract);
        auto graph = g.finish();
        graph->model_adapters = std::move(adapters);
        write_mmproj_config(*graph, ctx.metadata, config);
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
