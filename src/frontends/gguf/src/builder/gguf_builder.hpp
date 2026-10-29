// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "node_context.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

struct GgufGraph;  // defined in gguf_graph.hpp; only used here as a shared_ptr return type
class ArchRegistry;

using GraphBuilder = std::function<std::shared_ptr<GgufGraph>(const std::unordered_map<std::string, CreatorFunction>&)>;
// Parse the file and select a definition; invoke the builder with the converters active at conversion.
GraphBuilder load_gguf_builder(const std::filesystem::path& file,
							  const ArchRegistry& registry,
							  const GGUFLoadOptions& options = {});
GraphBuilder load_gguf_builder(const std::vector<std::filesystem::path>& files,
							  const ArchRegistry& registry,
							  const GGUFLoadOptions& options = {});

}  // namespace ov::frontend::gguf
