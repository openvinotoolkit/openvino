// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph_functions/subgraph_builders.hpp>

#include "ie_common.h"
#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

#include "functional_test_utils/plugin_cache.hpp"

using config_t = std::map<std::string, std::string>;

typedef myriadLayerTestBaseWithParam<config_t> myriadCorrectModelsConfigsTests_nightly;
typedef myriadLayerTestBaseWithParam<config_t> myriadIncorrectModelsConfigsTests_nightly;

//------------------------------------------------------------------------------
//  myriadCorrectModelsConfigsTests_nightly
//------------------------------------------------------------------------------

TEST_P(myriadCorrectModelsConfigsTests_nightly, LoadNetworkWithCorrectConfig) {
    const auto &config = GetParam();
    DISABLE_IF(!hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::ExecutableNetwork executable;
    ASSERT_NO_THROW(executable = _vpuPluginPtr->LoadNetwork(net, config));
}

TEST_P(myriadCorrectModelsConfigsTests_nightly, CreateInferRequestWithAvailableDevice) {
    const auto &config = GetParam();
    DISABLE_IF(!hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::ExecutableNetwork executable;
    ASSERT_NO_THROW(executable = _vpuPluginPtr->LoadNetwork(net, config));

    InferenceEngine::InferRequest request;
    ASSERT_NO_THROW(request = executable.CreateInferRequest());
}

//------------------------------------------------------------------------------
//  myriadIncorrectModelsConfigsTests_nightly
//------------------------------------------------------------------------------

TEST_P(myriadIncorrectModelsConfigsTests_nightly, LoadNetworkWithIncorrectConfig) {
    const auto &config = GetParam();

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::ExecutableNetwork executable;
    ASSERT_THROW(executable = _vpuPluginPtr->LoadNetwork(net, config),
        InferenceEngine::Exception);
}

//------------------------------------------------------------------------------
//  Tests initiation
//------------------------------------------------------------------------------

static const std::vector<config_t> myriadCorrectPlatformConfigValues = {
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), ""}}
};

static const std::vector<config_t> myriadIncorrectPlatformConfigValues = {
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), " 0"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "MyriadX"}}
};

static const std::vector<config_t> myriadCorrectPackageTypeConfigValues = {
    // Please do not use other types of DDR in tests with a real device, because it may hang.
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}}
};

static const std::vector<config_t> myriadIncorrectPackageTypeConfigValues = {
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-1"}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-MICRON_1GB"}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-MICRON_1GB"}},
};

INSTANTIATE_TEST_SUITE_P(MyriadConfigs, myriadCorrectModelsConfigsTests_nightly,
                        ::testing::ValuesIn(myriadCorrectPlatformConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadConfigs, myriadIncorrectModelsConfigsTests_nightly,
                        ::testing::ValuesIn(myriadIncorrectPlatformConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadPackageConfigs, myriadCorrectModelsConfigsTests_nightly,
    ::testing::ValuesIn(myriadCorrectPackageTypeConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadPackageConfigs, myriadIncorrectModelsConfigsTests_nightly,
    ::testing::ValuesIn(myriadIncorrectPackageTypeConfigValues));
