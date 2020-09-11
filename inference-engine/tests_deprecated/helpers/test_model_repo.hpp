// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

std::string get_model_repo();

namespace TestDataHelpers {

const char *getModelPathNonFatal() noexcept;

std::string get_data_path();

inline const char *getModelPathNonFatalDefault() noexcept {
    if (const auto envVar = std::getenv("MODELS_PATH")) {
        return envVar;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return nullptr;
#endif
};

inline std::string get_data_path_default() {
    if (const auto envVar = std::getenv("DATA_PATH")) {
        return envVar;
    }

#ifdef DATA_PATH
    return DATA_PATH;
#else
    return nullptr;
#endif
}
}  // namespace TestDataHelpers
