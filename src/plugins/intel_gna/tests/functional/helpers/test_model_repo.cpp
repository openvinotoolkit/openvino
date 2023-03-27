// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

std::string GetModelRepo() {
    return "models:";
};

const char* TestDataHelpers::GetModelPathNonFatal() noexcept {
    return TestDataHelpers::GetModelPathNonFatalDefault();
}

std::string TestDataHelpers::GetDataPath() {
    return TestDataHelpers::GetDataPathDefault();
}
