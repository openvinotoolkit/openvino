// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

std::string get_model_repo();

namespace TestDataHelpers {

const char* get_model_path_non_fatal() noexcept;

std::string get_data_path();

inline const char* get_model_path_non_fatal_default() noexcept {
    if (const auto env_var = std::getenv("MODELS_PATH")) {
        return env_var;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return nullptr;
#endif
};

inline std::string get_data_path_default() {
    if (const auto env_var = std::getenv("GNA_DATA_PATH")) {
        return env_var;
    }

#ifdef GNA_DATA_PATH
    return GNA_DATA_PATH;
#else
    return nullptr;
#endif
}
}  // namespace TestDataHelpers
