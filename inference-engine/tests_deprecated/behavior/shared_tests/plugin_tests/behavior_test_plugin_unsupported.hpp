// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
    std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
        return obj.param.device + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "") +
               "_" + getModelName(obj.param.model_xml_str) + "_" + obj.param.input_blob_precision.name();
    }
}

// Load unsupported network type to the Plugin
TEST_P(BehaviorPluginTestAllUnsupported, cannotLoadUnsupportedNetwork) {
    std::string refError = "The plugin does not support";
    auto param = GetParam();
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);
    ASSERT_THROW(core.LoadNetwork(network, param.device, param.config), InferenceEngineException);
}

// Load incorrect input type for Plugin
TEST_P(BehaviorPluginTestTypeUnsupported, LoadIncorrectInputType) {
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    inputs.begin()->second->setPrecision(GetParam().input_blob_precision);
    ASSERT_THROW(core.LoadNetwork(network, GetParam().device, GetParam().config), InferenceEngineException);
}

TEST_P(BehaviorPluginTestBatchUnsupported, DISABLED_LoadIncorrectBatchSize) {
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    network.setBatchSize(GetParam().batch_size);
    ASSERT_THROW(core.LoadNetwork(network, GetParam().device, GetParam().config), InferenceEngineException);
}
