// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Parse GGUF, select an architecture definition, and build a graph for the shared op translators.
// See docs/adding_an_architecture.md for the builder layers and registration paths.

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
#include "openvino/util/log.hpp"
#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

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

}  // namespace

GraphBuilder load_gguf_builder(const std::string& file, const ArchRegistry& registry) {
    auto data = std::make_shared<decltype(get_gguf_data(file))>(get_gguf_data(file));
    auto& [metadata, weights, qtypes, mmap, quant_buf] = *data;

    const detail::MetadataStore meta_store{metadata};
    const GgufMetadata meta_view(meta_store);

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
    if (definition->maturity == Maturity::Experimental) {
        OPENVINO_WARN("[GGUF] architecture handler '",
                      definition->id,
                      "' is experimental; validate accuracy before relying on it.");
    }
    return [data, definition = *definition](const std::unordered_map<std::string, CreatorFunction>& translators) {
        const auto& [metadata, source_weights, source_qtypes, mmap, quant_buf] = *data;
        auto weights = source_weights;
        auto qtypes = source_qtypes;
        const detail::MetadataStore meta_store{metadata};
        const GgufMetadata meta_view(meta_store);
        detail::WeightStore weight_store{weights, qtypes, &translators};
        BuildContext ctx{meta_view, meta_view.architecture(), &weight_store};
        auto builder = definition.factory(ctx);
        OPENVINO_ASSERT(builder, "[GGUF] architecture handler '", definition.id, "' returned no builder");
        auto graph = builder->build();
        OPENVINO_ASSERT(graph, "[GGUF] architecture handler '", definition.id, "' returned no graph");
        graph->tokenizer_config = extract_tokenizer_config(metadata);
        return graph;
    };
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
