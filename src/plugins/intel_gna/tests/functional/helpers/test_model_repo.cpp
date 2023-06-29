// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

std::string get_model_repo() {
    return "models:";
};

const char* TestDataHelpers::get_model_path_non_fatal() noexcept {
    return TestDataHelpers::get_model_path_non_fatal_default();
}

std::string TestDataHelpers::get_data_path() {
    return TestDataHelpers::get_data_path_default();
}
