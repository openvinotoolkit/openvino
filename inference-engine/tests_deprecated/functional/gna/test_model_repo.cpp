// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

std::string get_model_repo() {
    return "models:";
};

const char* TestDataHelpers::getModelPathNonFatal() noexcept {
    return TestDataHelpers::getModelPathNonFatalDefault();
}

std::string TestDataHelpers::get_data_path() {
    return TestDataHelpers::get_data_path_default();
}