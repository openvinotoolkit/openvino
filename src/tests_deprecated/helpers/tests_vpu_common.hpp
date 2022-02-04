// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <gtest/gtest.h>

#include <ie_precision.hpp>

#include "single_layer_common.hpp"
#include <vpu/private_plugin_config.hpp>


using config_t = std::map<std::string, std::string>;

static constexpr char ENV_MYRIADX[] = "IE_VPU_MYRIADX";
static constexpr char ENV_HDDL_R[]  = "IE_VPU_ENABLE_PER_LAYER_TESTS_HDDL";

#define DISABLE_IF(expression)                                   \
{                                                                \
    if (expression) {                                            \
        GTEST_SKIP() << "Disabled since " << #expression << std::endl; \
    }                                                            \
}

#ifdef _WIN32
    #define DISABLE_ON_WINDOWS_IF(expr) DISABLE_IF((expr))
#else
    #define DISABLE_ON_WINDOWS_IF(expr)
#endif

static bool hasPlatform(const std::string &environment_variable) {
    auto env = std::getenv(environment_variable.c_str());
    if (!env) {
        return false;
    }

    int value;
    try {
        value = std::stoi(env);
    } catch (...) {
        return false;
    }

    return value != 0;
}

static bool hasMyriadX() {
    return hasPlatform(ENV_MYRIADX);
}

static bool hasMyriad2() {
    /* TODO: change with environment variable for MYRIAD-2 */
    return !hasMyriadX();
}

static bool hasAppropriateStick(const config_t &config) {
    bool suitsConfig;

    bool hasRequestedMyriadX = hasMyriadX();
    suitsConfig = hasRequestedMyriadX;

    return suitsConfig;
}

static bool hasHDDL_R() {
    return hasPlatform(ENV_HDDL_R);
}
