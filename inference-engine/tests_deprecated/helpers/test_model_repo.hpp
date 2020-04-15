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
#ifdef MODELS_PATH
    const char *models_path = std::getenv("MODELS_PATH");

    if (models_path == nullptr && MODELS_PATH == nullptr) {
        return nullptr;
    }

    if (models_path == nullptr) {
        return MODELS_PATH;
    }

    return models_path;
#else
    return nullptr;
#endif
};

inline std::string get_data_path_default() {
#ifdef DATA_PATH
    const char *data_path = std::getenv("DATA_PATH");

    if (data_path == NULL) {
        if (DATA_PATH != NULL) {
            data_path = DATA_PATH;
        } else {
            return nullptr;
        }
    }
    return std::string(data_path);
#else
    return nullptr;
#endif
}
}  // namespace TestDataHelpers
