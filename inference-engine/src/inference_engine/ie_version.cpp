// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_version.hpp"

namespace InferenceEngine {

INFERENCE_ENGINE_API(const Version*) GetInferenceEngineVersion() noexcept {
    // Use local static variable to make sure it is always properly initialized
    // even if called from global constructor
    static Version inferenceEngineVersion = {
        {1, 2},  // inference engine API version
        CI_BUILD_NUMBER
    };
    return &inferenceEngineVersion;
}
}  // namespace InferenceEngine
