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

// Parse a .gguf file and invoke its registered builder without a llama.cpp dependency.
// Built-in and external definitions share dispatch; throws if no definition matches.
std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file, const ArchRegistry& registry);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
