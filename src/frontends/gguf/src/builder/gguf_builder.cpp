// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Entry point of the native GGUF -> GgufGraph path: parse the container, decide which model FAMILY
// the file holds, and hand it to that family's ModelBuilder. No llama.cpp / gguf dependency.
//
// The emitted nodes use the GGML op vocabulary and reproduce llama.cpp's cgraph topology, so the
// resulting GgufGraph drives the same op translators as the llama.cpp cgraph path.
//
// Layering (see docs/adding_an_architecture.md):
//   graph_emitter.hpp     arch-agnostic "how do I write a node"
//   blocks/               reusable graph fragments (norm, ffn, attention, gated delta net)
//   decoder_config.hpp    all per-architecture detection for the decoder family
//   arch/decoder_builder  the decoder family's topology
//   arch_registry.hpp     which architectures are accepted, and their RoPE mode
//   model_kind.hpp        which family a file belongs to
//
// Adding an ARCHITECTURE of an existing family is a name in arch_registry.cpp.
// Adding a FAMILY (mmproj vision/audio, encoder-decoder) is a new ModelBuilder subclass plus a
// branch in build_ggml_graph_from_gguf below; nothing existing changes.

#include "gguf_builder.hpp"

#include <memory>
#include <type_traits>

#include "builder/arch/decoder_builder.hpp"
#include "builder/arch_registry.hpp"
#include "builder/model_builder.hpp"
#include "builder/model_kind.hpp"
#include "gguf_graph.hpp"
#include "openvino/core/except.hpp"
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

std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes, mmap, quant_buf] = get_gguf_data(file);

    // Decide the family FIRST: the metadata key layout differs per family, so reading any
    // decoder hyperparameter before this point would misreport an mmproj file as a broken LLM.
    const ModelKind kind = detect_model_kind(metadata);
    OPENVINO_ASSERT(kind == ModelKind::Decoder,
                    "[GGUF] this file holds a ",
                    model_kind_name(kind),
                    " model; the native GGUF builder currently implements the decoder family only. "
                    "Support is added by implementing a ModelBuilder subclass for that family (see "
                    "builder/model_builder.hpp).");

    auto config = decoder_config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(supported_archs().count(arch),
                    "[GGUF] native GGUF builder does not support architecture '",
                    arch,
                    "'. See supported_archs() in builder/arch_registry.cpp for the full list.");
    if (experimental_archs().count(arch)) {
        OPENVINO_WARN("[GGUF] architecture '",
                      arch,
                      "' is experimental: it is built by structural auto-detection but has not been "
                      "end-to-end verified against a reference. Validate accuracy before relying on it.");
    }

    std::unique_ptr<ModelBuilder> builder = std::make_unique<DecoderBuilder>(config, weights, qtypes);
    auto graph = builder->build();
    graph->tokenizer_config = extract_tokenizer_config(metadata);
    return graph;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
