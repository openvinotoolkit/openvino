// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <map>
#include <gtest/gtest.h>
#include "behavior_test_plugin.h"
#include "helpers/myriad_load_network_case.hpp"

TEST_F(MyriadLoadNetworkTestCase, smoke_ReloadPlugin) {
    ASSERT_NO_THROW(LoadNetwork());
    ASSERT_NO_THROW(LoadNetwork());
}

TEST_F(MyriadLoadNetworkTestCase, smoke_SimpleLoading) {
    auto devices = getDevicesList();
    ASSERT_TRUE(devices.size());

    auto device_to_load = devices[0];
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_NO_THROW(ExeNetwork exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));

    ASSERT_TRUE(!IsDeviceAvailable(device_to_load));
}

TEST_F(MyriadLoadNetworkTestCase, smoke_LoadingAtTheSameDevice) {
    auto devices = getDevicesList();
    ASSERT_TRUE(devices.size());

    auto device_to_load = devices[0];
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_NO_THROW(ExeNetwork exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));

    ASSERT_TRUE(!IsDeviceAvailable(device_to_load));

    ASSERT_NO_THROW(ExeNetwork exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));
}

TEST_F(MyriadLoadNetworkTestCase, smoke_ThrowsExeptionWhenNameIsInvalid) {
    auto device_to_load = "SomeVeryBadName";
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_ANY_THROW(ExeNetwork exe_network =
        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));
}
