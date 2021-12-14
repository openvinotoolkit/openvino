// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <chrono>
#include <gtest/gtest.h>
#include <ie_blob.h>
#include <string>
#include <precision_utils.h>
#include <tests_common.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include "vpu_case_params.hpp"
#include "vpu_param_containers.hpp"

using namespace ::testing;
using namespace InferenceEngine;

#define DISABLE_IF(expr) \
    do { \
        if (expr) { \
            GTEST_SKIP() << "Disabled since " << #expr << std::endl; \
        } \
    }while(false)


#ifdef _WIN32
#   define DISABLE_ON_WINDOWS_IF(expr) DISABLE_IF((expr))
#else
#   define DISABLE_ON_WINDOWS_IF(expr)
#endif

#if defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || defined(_M_ARM64)
#   define DISABLE_ON_ARM      GTEST_SKIP() << "Disabled on ARM" << std::endl;
#   define VPU_REG_TEST_ARM_PLATFORM
#else
#   define DISABLE_ON_ARM
#endif

#define ENABLE_IF_MA2085 \
    do { \
        if (!CheckMA2085()) { \
            GTEST_SKIP() << "Disabled since not on MA2085" << std::endl; \
        }\
    } while(false)

extern bool CheckMyriadX();
extern bool CheckMA2085();

//------------------------------------------------------------------------------
// Parameters definition
//------------------------------------------------------------------------------

using Batch = int;
using DoReshape = bool;
using Resources = int;
using PluginDevicePair = std::pair<std::string, std::string>;

//------------------------------------------------------------------------------
// class VpuNoRegressionBase
//------------------------------------------------------------------------------

class VpuNoRegressionBase : public TestsCommon {
public:
    //Operations
    static std::string getTestCaseName(PluginDevicePair,
                                       Precision,
                                       Batch,
                                       DoReshape);

    // Accessors
    std::string getDeviceName() const;

protected:
    // Data section
    std::string plugin_name_;
    std::string device_name_;
    Precision in_precision_;
    int batch_;
    bool do_reshape_;
    std::map <std::string, std::string> config_;

    //Operations
    virtual void InitConfig();
};
