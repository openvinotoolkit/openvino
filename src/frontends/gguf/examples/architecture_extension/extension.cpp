// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ArchitectureExtension>(example::projector_architecture())});
