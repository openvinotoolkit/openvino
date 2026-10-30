// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "builder/gguf_graph.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {
std::shared_ptr<ModelBuilder> make_mmproj_builder(const BuildContext& context);

// Builds every declared clip encoder as `<modality>.embeddings` and returns the built modalities.
std::vector<std::string> build_mmproj_encoders(GgufGraphContext& g,
                                               const BuildContext& ctx,
                                               ModelAdapters& adapters,
                                               std::map<std::string, std::string>& config);
// Stores clip.* metadata followed by `config` in the graph's mmproj rt_info.
void write_mmproj_config(GgufGraph& graph,
                         const GgufMetadata& metadata,
                         const std::map<std::string, std::string>& config);
}  // namespace ov::frontend::gguf
