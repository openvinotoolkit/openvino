// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_version.hpp"

namespace InferenceEngine {

#if defined(__ANDROID__)
#define CI_BUILD_NUMBER "custom-master-android-nn"
#endif
const Version* GetInferenceEngineVersion() noexcept {
    // Use local static variable to make sure it is always properly initialized
    // even if called from global constructor
    static Version inferenceEngineVersion = {{2, 1},  // inference engine API version
                                             CI_BUILD_NUMBER,
                                             "API"};
    return &inferenceEngineVersion;
}
}  // namespace InferenceEngine
