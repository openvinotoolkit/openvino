// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector.hpp"

namespace {
std::vector<ov::Extension::Ptr> make_projector_extensions() {
    const auto architecture =
        std::make_shared<ov::frontend::gguf::ArchitectureExtension>(example::projector_architecture());
    const auto projector =
        std::make_shared<ov::frontend::gguf::ProjectorExtension>(example::vision_projector_component());
    return {architecture, projector};
}
}  // namespace

OPENVINO_CREATE_EXTENSIONS(make_projector_extensions());
