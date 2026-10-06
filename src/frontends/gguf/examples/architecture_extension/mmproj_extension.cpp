// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"

namespace {
ov::frontend::gguf::ProjectorDefinition projector() {
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
}  // namespace

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ProjectorExtension>(projector())})
