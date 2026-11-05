// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"

namespace {
ov::frontend::gguf::ProjectorDefinition gemma3_alias() {
    using namespace ov::frontend::gguf;
    return {"clip.vision.example-gemma3",
            "clip",
            "vision",
            "example-gemma3",
            [](GgufGraphContext& graph) {
                return build_builtin_projector(graph, "vision", "gemma3");
            },
            {}};
}
ov::frontend::gguf::ProjectorDefinition audio_projection() {
    using namespace ov::frontend::gguf;
    return {"clip.audio.example-linear",
            "clip",
            "audio",
            "example-linear",
            [](GgufGraphContext& graph) {
                const auto weight = graph.tensors().require("example.audio.projection.weight");
                const auto input =
                    graph.add_input("audio.example_embeddings", ov::element::f32, {1, 1, -1, weight.ne(0)});
                return ProjectorResult{graph.node("GGML_OP_MUL_MAT", {weight, input}), {{"audio.merge", "1"}}};
            },
            {}};
}
}  // namespace

OPENVINO_CREATE_EXTENSIONS((std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ProjectorExtension>(gemma3_alias()),
    std::make_shared<ov::frontend::gguf::ProjectorExtension>(audio_projection())}))
