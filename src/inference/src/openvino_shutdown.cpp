// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/core.hpp"

void ov::shutdown() {
    frontend::FrontEndManager::shutdown();
}

void InferenceEngine::shutdown() {
    ov::shutdown();
}
