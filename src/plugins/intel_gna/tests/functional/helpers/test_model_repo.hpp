// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

std::string GetModelRepo();

namespace TestDataHelpers {

const char* GetModelPathNonFatal() noexcept;

std::string GetDataPath();

inline const char* GetModelPathNonFatalDefault() noexcept {
    if (const auto env_var = std::getenv("MODELS_PATH")) {
        return env_var;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return nullptr;
#endif
};

inline std::string GetDataPathDefault() {
    if (const auto env_var = std::getenv("DATA_PATH")) {
        return env_var;
    }

#ifdef DATA_PATH
    return DATA_PATH;
#else
    return nullptr;
#endif
}
}  // namespace TestDataHelpers
