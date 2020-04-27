// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <map>
#include <gtest/gtest.h>
#include "behavior_test_plugin.h"
#include "helpers/myriad_load_network_case.hpp"

TEST_F(MyriadLoadNetworkTestCase, ReloadPlugin) {
    ASSERT_NO_THROW(LoadNetwork());
    ASSERT_NO_THROW(LoadNetwork());
}

TEST_F(MyriadLoadNetworkTestCase, SimpleLoading) {
    auto devices = getDevicesList();
    ASSERT_TRUE(devices.size());

    auto device_to_load = devices[0];
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_NO_THROW(ExeNetworkPtr exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));

    ASSERT_TRUE(!IsDeviceAvailable(device_to_load));
}

TEST_F(MyriadLoadNetworkTestCase, LoadingAtTheSameDevice) {
    auto devices = getDevicesList();
    ASSERT_TRUE(devices.size());

    auto device_to_load = devices[0];
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_NO_THROW(ExeNetworkPtr exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));

    ASSERT_TRUE(!IsDeviceAvailable(device_to_load));

    ASSERT_NO_THROW(ExeNetworkPtr exe_network =
                        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));
}

TEST_F(MyriadLoadNetworkTestCase, ThrowsExeptionWhenNameIsInvalid) {
    auto device_to_load = "SomeVeryBadName";
    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
    };

    ASSERT_ANY_THROW(ExeNetworkPtr exe_network =
        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));
}

TEST_F(MyriadLoadNetworkTestCase, ThrowsExeptionWhenPlatformConflictWithProtocol) {
    std::string wrong_platform;
    auto devices = getDevicesList();
    ASSERT_TRUE(devices.size());

    auto device_to_load = devices[0];

    IE_SUPPRESS_DEPRECATED_START
    if(isMyriadXDevice(device_to_load)) {
        wrong_platform = VPU_MYRIAD_2450;
    } else {
        wrong_platform = VPU_MYRIAD_2480;
    }
    IE_SUPPRESS_DEPRECATED_END

    std::map<std::string, std::string> config = {
        {KEY_DEVICE_ID, device_to_load},
        {KEY_VPU_MYRIAD_PLATFORM, wrong_platform},
    };

    ASSERT_ANY_THROW(ExeNetworkPtr exe_network =
        ie->LoadNetwork(cnnNetwork, "MYRIAD", config));
}
