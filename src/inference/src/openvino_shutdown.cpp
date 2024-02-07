// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/manager.hpp"
#include "openvino/runtime/core.hpp"

void ov::shutdown() {
    frontend::FrontEndManager::shutdown();
}
