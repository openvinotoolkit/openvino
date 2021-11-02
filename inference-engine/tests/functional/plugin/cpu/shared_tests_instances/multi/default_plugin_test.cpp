// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "ie_core.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "base/multi/multi_helpers.hpp"

const std::vector<DevicesNames> device_names {
        {""}  // use default device in ie core
};
TEST_P(MultiDevice_Test, canLoadDefaultAutoPluginTest) {
    InferenceEngine::CNNNetwork net(fn_ptr);

    auto ie = PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(net, ""));
    InferenceEngine::Parameter p;
    ASSERT_NO_THROW(p = execNet.GetConfig(MULTI_CONFIG_KEY(DEVICE_PRIORITIES)));
}
INSTANTIATE_TEST_SUITE_P(smoke_DefaultPluginAuto, MultiDevice_Test,
        ::testing::ValuesIn(device_names), MultiDevice_Test::getTestCaseName);
