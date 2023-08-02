// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/env_util.hpp"

#include "openvino/util/env_util.hpp"

std::string ngraph::getenv_string(const char* env_var) {
    return ov::util::getenv_string(env_var);
}

int32_t ngraph::getenv_int(const char* env_var, int32_t default_value) {
    return ov::util::getenv_int(env_var, default_value);
}

bool ngraph::getenv_bool(const char* env_var, bool default_value) {
    return ov::util::getenv_bool(env_var, default_value);
}
