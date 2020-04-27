// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <array>

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "")
           + "_" + obj.param.input_blob_precision.name();
}
}

// Load unsupported network type to the Plugin
TEST_P(BehaviorPluginTestVersion, pluginCurrentVersionIsCorrect) {
    std::string refError = "The plugin does not support";
    InferenceEngine::Core core;
    const std::string device = GetParam().device;
    if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        std::map<std::string, InferenceEngine::Version> versions = core.GetVersions(GetParam().device);
        ASSERT_EQ(versions.size(), 1);
        auto version = versions.begin()->second;
        ASSERT_EQ(version.apiVersion.major, 2);
        ASSERT_EQ(version.apiVersion.minor, 1);
    }
}

template <typename T, size_t N>
std::array<T, N+1> add_element_into_array(const T (&arr)[N], const T & element) {
    std::array<T, N+1> ar;
    for(size_t i =  0; i != N; i++) {
        ar[i] = arr[i];
    }
    ar[N] = element;
    return ar;

};
