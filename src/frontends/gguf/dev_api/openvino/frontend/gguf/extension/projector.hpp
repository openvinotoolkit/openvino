// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <optional>

#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/model_adapter.hpp"

namespace ov::frontend::gguf {

// Recipes use resolved vision.patch_size/window_size/merge and clip.* dimensions in config.
enum class ProjectorLayout { Unsupported, Image, PaddedPatches, WindowedPatches, SplitMerger, PackedPatches };

struct GGUF_FRONTEND_API ProjectorResult {
    GgufValue output;
    std::map<std::string, std::string> config;
    // Overrides the definition; Image keeps the inputs supplied by build().
    std::optional<ProjectorLayout> genai_layout;
    // Overrides the definition; an explicit empty callback selects the serializable recipe.
    std::optional<ModelAdapterFn> genai_adapter;
};

struct GGUF_FRONTEND_API ProjectorDefinition {
    using BuildFn = std::function<ProjectorResult(GgufGraphContext&)>;

    std::string id;
    std::string architecture;
    std::string modality;
    std::string projector_type;
    BuildFn build;
    ArchitectureDefinition::MatchFn match;
    std::optional<ProjectorLayout> genai_layout;
    ModelAdapterFn genai_adapter;
};

class GGUF_FRONTEND_API ProjectorExtension : public ArchitectureExtension {
public:
    OPENVINO_RTTI("gguf::ProjectorExtension", "", ArchitectureExtension);
    using Ptr = std::shared_ptr<ProjectorExtension>;
    explicit ProjectorExtension(ProjectorDefinition definition, RegistrationMode mode = RegistrationMode::Add);
    ~ProjectorExtension() override;

    const ProjectorDefinition& projector_definition() const {
        return m_projector;
    }

private:
    ProjectorDefinition m_projector;
};

// Append one built-in encoder/projector branch; the caller owns outputs and finish().
GGUF_FRONTEND_API ProjectorResult build_builtin_projector(GgufGraphContext& graph,
                                                          const std::string& modality,
                                                          const std::string& projector_type);

}  // namespace ov::frontend::gguf
