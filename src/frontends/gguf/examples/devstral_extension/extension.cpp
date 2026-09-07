// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "devstral.hpp"

using ov::frontend::gguf::ArchitectureExtension;
using ov::frontend::gguf::RegistrationMode;

OPENVINO_CREATE_EXTENSIONS((std::vector<ov::Extension::Ptr>{
    std::make_shared<ArchitectureExtension>(example::devstral_decoder("llama"), RegistrationMode::Replace),
    std::make_shared<ArchitectureExtension>(example::devstral_decoder("mistral3"), RegistrationMode::Replace)}));
