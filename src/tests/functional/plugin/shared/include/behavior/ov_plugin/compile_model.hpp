// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#endif

namespace ov {
namespace test {
namespace behavior {

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassLoadNetworkTest = OVClassBaseTestP;

TEST_P(OVClassNetworkTestP, LoadNetworkCreateDefaultExecGraphResult) {
    auto ie = createCoreWithTemplate();
    auto net = ie.compile_model(actualNetwork, target_device);
    auto runtime_function = net.get_runtime_model();
    ASSERT_NE(nullptr, runtime_function);
    auto actual_parameters = runtime_function->get_parameters();
    auto actual_results = runtime_function->get_results();
    auto expected_parameters = actualNetwork->get_parameters();
    auto expected_results = actualNetwork->get_results();
    ASSERT_EQ(expected_parameters.size(), actual_parameters.size());
    for (std::size_t i = 0; i < expected_parameters.size(); ++i) {
        auto expected_element_type = expected_parameters[i]->get_output_element_type(0);
        auto actual_element_type = actual_parameters[i]->get_output_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_parameters[i]->get_output_shape(0);
        auto actual_shape = actual_parameters[i]->get_output_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
    ASSERT_EQ(expected_results.size(), actual_results.size());
    for (std::size_t i = 0; i < expected_results.size(); ++i) {
        auto expected_element_type = expected_results[i]->get_input_element_type(0);
        auto actual_element_type = actual_results[i]->get_input_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_results[i]->get_input_shape(0);
        auto actual_shape = actual_results[i]->get_input_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        OV_ASSERT_NO_THROW(ie.compile_model(simpleNetwork, target_device + "." + deviceIDs[0]));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".10"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".l0"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
