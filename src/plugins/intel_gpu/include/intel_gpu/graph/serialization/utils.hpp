// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include "ie/ie_common.h"

namespace cldnn {
static InferenceEngine::Layout layout_from_string(const std::string& name) {
    static const std::unordered_map<std::string, InferenceEngine::Layout> layouts = {
        { "ANY", InferenceEngine::Layout::ANY },
        { "NCHW", InferenceEngine::Layout::NCHW },
        { "NHWC", InferenceEngine::Layout::NHWC },
        { "NCDHW", InferenceEngine::Layout::NCDHW },
        { "NDHWC", InferenceEngine::Layout::NDHWC },
        { "OIHW", InferenceEngine::Layout::OIHW },
        { "GOIHW", InferenceEngine::Layout::GOIHW },
        { "OIDHW", InferenceEngine::Layout::OIDHW },
        { "GOIDHW", InferenceEngine::Layout::GOIDHW },
        { "SCALAR", InferenceEngine::Layout::SCALAR },
        { "C", InferenceEngine::Layout::C },
        { "CHW", InferenceEngine::Layout::CHW },
        { "HWC", InferenceEngine::Layout::HWC },
        { "HW", InferenceEngine::Layout::HW },
        { "NC", InferenceEngine::Layout::NC },
        { "CN", InferenceEngine::Layout::CN },
        { "BLOCKED", InferenceEngine::Layout::BLOCKED }
    };
    auto it = layouts.find(name);
    if (it != layouts.end()) {
        return it->second;
    }
    IE_THROW(NetworkNotRead) << "Unknown layout with name '" << name << "'";
}
}  // namespace cldnn
