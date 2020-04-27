// Copyright (C) 2018-2020 Intel Corporation
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
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();
    DISABLE_IF(!hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = _vpuPluginPtr->LoadNetwork(executable, net, config, &response);

    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;
}

TEST_P(myriadCorrectModelsConfigsTests_nightly, CreateInferRequestWithAvailableDevice) {
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();
    DISABLE_IF(!hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = _vpuPluginPtr->LoadNetwork(executable, net, config, &response);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;

    InferenceEngine::IInferRequest::Ptr request;
    sts = executable->CreateInferRequest(request, &response);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;
}

TEST_P(myriadCorrectModelsConfigsTests_nightly, CreateInferRequestWithUnavailableDevice) {
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();
    DISABLE_IF(hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = _vpuPluginPtr->LoadNetwork(executable, net, config, &response);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;

    InferenceEngine::IInferRequest::Ptr request;
    sts = executable->CreateInferRequest(request, &response);
    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, sts) << response.msg;
}

//------------------------------------------------------------------------------
//  myriadIncorrectModelsConfigsTests_nightly
//------------------------------------------------------------------------------

TEST_P(myriadIncorrectModelsConfigsTests_nightly, LoadNetworkWithIncorrectConfig) {
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();
    DISABLE_IF(hasAppropriateStick(config));

    InferenceEngine::CNNNetwork net(ngraph::builder::subgraph::makeSplitConvConcat());
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = _vpuPluginPtr->LoadNetwork(executable, net, config, &response);

    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, sts) << response.msg;
}

//------------------------------------------------------------------------------
//  Tests initiation
//------------------------------------------------------------------------------

static const std::vector<config_t> myriadCorrectPlatformConfigValues = {
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), ""}},
        // Deprecated
        {{VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2450)}},
        {{VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)}},
        {{VPU_CONFIG_KEY(PLATFORM), ""}}
};

static const std::vector<config_t> myriadIncorrectPlatformConfigValues = {
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), " 0"}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "MyriadX"}},
        // Deprecated
        {{VPU_CONFIG_KEY(PLATFORM), "-1"}},
        {{VPU_CONFIG_KEY(PLATFORM), " 0"}},
        {{VPU_CONFIG_KEY(PLATFORM), "MyriadX"}},
        // Deprecated key & value from current
        {{VPU_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
        {{VPU_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
        // Current key & deprecated value
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2450)}},
        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)}},

};

INSTANTIATE_TEST_CASE_P(MyriadConfigs, myriadCorrectModelsConfigsTests_nightly, ::testing::ValuesIn(myriadCorrectPlatformConfigValues));

INSTANTIATE_TEST_CASE_P(MyriadConfigs, myriadIncorrectModelsConfigsTests_nightly, ::testing::ValuesIn(myriadIncorrectPlatformConfigValues));
