// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

struct GgufGraph;  // defined in gguf_graph.hpp; only used here as a shared_ptr return type
class ArchRegistry;

// Build a GgufGraph natively from a .gguf file (no llama.cpp / gguf dependency).
// Parses the container, then dispatches to a builder that emits nodes in the GGML op vocabulary
// reproducing llama.cpp's cgraph topology for that architecture.
//
// `registry` decides which architectures are accepted. An ArchitectureExtension registered on it
// is consulted FIRST and may supply its own builder, which is how an architecture -- of any
// family, decoder or not -- is added without rebuilding the frontend.
//
// Throws if no builder claims the file.
std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file, const ArchRegistry& registry);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
