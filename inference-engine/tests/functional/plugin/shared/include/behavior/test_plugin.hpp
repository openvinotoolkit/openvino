// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include <common_test_utils/ngraph_test_utils.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <fstream>
#include <base/behavior_test_utils.hpp>
#include <common_test_utils/test_assertions.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace BehaviorTestsDefinitions {
using BehaviorTests = BehaviorTestsUtils::BehaviorTestsBasic;

void setInputNetworkPrecision(InferenceEngine::CNNNetwork &network, InferenceEngine::InputsDataMap &inputs_info,
                              InferenceEngine::Precision input_precision) {
    inputs_info = network.getInputsInfo();
    ASSERT_EQ(1u, inputs_info.size());
    inputs_info.begin()->second->setPrecision(input_precision);
}

void setOutputNetworkPrecision(InferenceEngine::CNNNetwork &network, InferenceEngine::OutputsDataMap &outputs_info,
                               InferenceEngine::Precision output_precision) {
    outputs_info = network.getOutputsInfo();
    ASSERT_EQ(outputs_info.size(), 1u);
    outputs_info.begin()->second->setPrecision(output_precision);
}

TEST_P(BehaviorTests, allocateNullBlob) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::TensorDesc tdesc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                                    InferenceEngine::NCHW);
    InferenceEngine::TBlob<float> blob(tdesc);
    ASSERT_NO_THROW(blob.allocate());
}

TEST_P(BehaviorTests, pluginDoesNotChangeOriginalNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto param = GetParam();

    InferenceEngine::CNNNetwork cnnNet(function);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));

    // compare 2 networks
    auto referenceNetwork = ngraph::builder::subgraph::makeConvPoolRelu();
    compare_functions(referenceNetwork, cnnNet.getFunction());
}

using BehaviorTestInput = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(BehaviorTestInput, canSetInputPrecisionForNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::InputsDataMap inputs_info;
    InferenceEngine::CNNNetwork cnnNet(function);
    setInputNetworkPrecision(cnnNet, inputs_info, netPrecision);

    // Input image format I16 is not supported yet.
    if (( targetDevice == CommonTestUtils::DEVICE_MYRIAD
            || targetDevice == CommonTestUtils::DEVICE_HDDL
            || targetDevice == CommonTestUtils::DEVICE_KEEMBAY)
         && netPrecision == InferenceEngine::Precision::I16) {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration), InferenceEngine::GeneralError);
    } else {
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}

using BehaviorTestOutput = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(BehaviorTestOutput, canSetOutputPrecisionForNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::OutputsDataMap output_info;
    InferenceEngine::CNNNetwork cnnNet(function);
    setOutputNetworkPrecision(cnnNet, output_info, netPrecision);
    if ((netPrecision == InferenceEngine::Precision::I16 || netPrecision == InferenceEngine::Precision::U8)) {
        if ((targetDevice == "CPU") || (targetDevice == "GPU"))  {
            ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
        } else {
            GTEST_SKIP();
        }
    } else {
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}
}  // namespace BehaviorTestsDefinitions