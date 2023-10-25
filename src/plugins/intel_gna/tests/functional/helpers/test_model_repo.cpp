// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

std::string TestDataHelpers::get_data_path() {
    if (const auto env_var = std::getenv("GNA_DATA_PATH")) {
        return env_var;
    }

#ifdef GNA_DATA_PATH
    return GNA_DATA_PATH;
#else
    return nullptr;
#endif
}
