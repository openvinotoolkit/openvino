// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Parse GGUF, select an architecture definition, and build a graph for the shared op translators.
// See docs/architectures.md for the builder layers and registration paths.

#include "gguf_builder.hpp"

#include <memory>
#include <type_traits>

#include "builder/api/metadata_store.hpp"
#include "builder/arch/decoder_builder.hpp"
#include "builder/arch_registry.hpp"
#include "builder/model_kind.hpp"
#include "gguf_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

namespace {

// Pull the `tokenizer.*` ggml metadata into an ov::AnyMap keyed by the sub-key after the last
// dot (e.g. "tokenizer.ggml.tokens" -> "tokens", "tokenizer.chat_template" -> "chat_template").
// Each GGUF metadata variant is mapped to the ov::Any types a downstream tokenizer builder
// consumes: std::string / std::vector<std::string> / ov::Tensor (arrays and shape-{} scalars).
ov::AnyMap extract_tokenizer_config(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    const std::string prefix = "tokenizer.";
    ov::AnyMap cfg;
    for (const auto& [key, value] : metadata) {
        if (key.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const auto sub_key = key.substr(key.find_last_of('.') + 1);
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    // skip empty
                } else {
                    cfg[sub_key] = v;
                }
            },
            value);
    }
    return cfg;
}

GraphBuilder make_gguf_builder(GGUFLoad loaded,
                               const ArchRegistry& registry,
                               const std::string& component,
                               bool export_hidden_state) {
    auto data = std::make_shared<GGUFLoad>(std::move(loaded));
    auto& metadata = std::get<0>(*data);

    const detail::MetadataStore meta_store{metadata};
    const GgufMetadata meta_view(meta_store);
    if (meta_view.architecture() == "this model cannot be used as LLM, use it via --model-vocoder in TTS examples") {
        metadata["general.architecture"] = std::string("lfm2.audio_vocoder");
    }

    // Built-in and external definitions share selection, construction and postprocessing.
    // Family detection is only a diagnostic fallback; it must not reject a registered family.
    const auto definition = registry.find(meta_view);
    if (!definition) {
        const auto kind = detect_model_kind(metadata);
        OPENVINO_ASSERT(kind == ModelKind::DECODER,
                        "[GGUF] no builder for ",
                        model_kind_name(kind),
                        "; the default catalog implements the decoder family. Register an ArchitectureExtension.");
        OPENVINO_THROW("[GGUF] native GGUF builder does not support architecture '",
                       meta_view.architecture(),
                       "'. Supported: ",
                       registry.describe_supported(),
                       ". Register an ArchitectureExtension.");
    }
    return [data, definition = *definition, projectors = registry.projectors(), component, export_hidden_state](
               const std::unordered_map<std::string, CreatorFunction>& translators) {
        const auto& [metadata, weights, mmap] = *data;
        const detail::MetadataStore meta_store{metadata};
        const GgufMetadata meta_view(meta_store);
        detail::WeightStore weight_store{weights, &translators};
        BuildContext ctx{meta_view, meta_view.architecture(), &weight_store, &projectors};
        ctx.component = component;
        ctx.export_hidden_state = export_hidden_state;
        auto builder = definition.factory(ctx);
        OPENVINO_ASSERT(builder, "[GGUF] architecture handler '", definition.id, "' returned no builder");
        auto graph = builder->build();
        OPENVINO_ASSERT(graph, "[GGUF] architecture handler '", definition.id, "' returned no graph");
        OPENVINO_ASSERT(graph->component == component,
                        "[GGUF] architecture handler '",
                        definition.id,
                        "' does not implement component '",
                        component,
                        "'");
        graph->tokenizer_config = extract_tokenizer_config(metadata);
        graph->architecture = meta_view.architecture();
        return graph;
    };
}

}  // namespace

GraphBuilder load_gguf_builder(const std::filesystem::path& file,
                               const ArchRegistry& registry,
                               const std::string& component,
                               bool export_hidden_state) {
    return make_gguf_builder(get_gguf_data(file), registry, component, export_hidden_state);
}

GraphBuilder load_gguf_builder(const std::vector<std::filesystem::path>& files,
                               const ArchRegistry& registry,
                               const std::string& component,
                               bool export_hidden_state) {
    return make_gguf_builder(get_gguf_data(files), registry, component, export_hidden_state);
}

}  // namespace ov::frontend::gguf
