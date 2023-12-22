// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_version.hpp"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {

const Version* GetInferenceEngineVersion() noexcept {
    // Use local static variable to make sure it is always properly initialized
    // even if called from global constructor
    static Version inferenceEngineVersion = {{2, 1},  // inference engine API version
                                             CI_BUILD_NUMBER,
                                             "IE"};
    return &inferenceEngineVersion;
}
}  // namespace InferenceEngine
