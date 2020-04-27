// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "behavior_test_plugin.h"
#include "details/ie_cnn_network_tools.h"
#include "exec_graph_info.hpp"

using namespace ::testing;
using namespace InferenceEngine;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name()
           + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

inline std::vector<std::string> separateStrToVec(std::string str, const char sep) {
    std::vector<std::string> result;

    std::istringstream stream(str);
    std::string strVal;

    while (getline(stream, strVal, sep)) {
        result.push_back(strVal);
    }
    return result;
}


TEST_P(BehaviorPluginTestExecGraphInfo, CheckExecGraphInfoBeforeExecution) {
    auto param = GetParam();

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));

    auto cnnNetwork = testEnv->network;
    auto exeNetwork = testEnv->exeNetwork;

    if (param.device == CommonTestUtils::DEVICE_CPU || param.device == CommonTestUtils::DEVICE_GPU) {
        CNNNetwork execGraph;
        ASSERT_NO_THROW(execGraph = exeNetwork.GetExecGraphInfo());

        // Store all the original layers from the network
        const std::vector<CNNLayerPtr> originalLayers = CNNNetSortTopologically(cnnNetwork);
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            originalLayersMap[layer->name] = 0;
        }

        // Store all the layers from the executable graph information represented as CNNNetwork
        const std::vector<CNNLayerPtr> execGraphLayers = CNNNetSortTopologically(execGraph);
        for (const auto &execLayer : execGraphLayers) {
            // Each layer from the execGraphInfo network must have PM data option set
            ASSERT_EQ("not_executed", execLayer->params[ExecGraphInfoSerialization::PERF_COUNTER]);

            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = execLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');

            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            } );
        }
        // All layers from the original IR must be present within ExecGraphInfo
        for (auto& layer : originalLayersMap) {
            ASSERT_GT(layer.second, 0);
        }
    } else {
        // Not implemented for other plugins
        ASSERT_THROW(exeNetwork.GetExecGraphInfo(), InferenceEngineException);
    }
}

TEST_P(BehaviorPluginTestExecGraphInfo, CheckExecGraphInfoAfterExecution) {
    auto param = GetParam();

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv,
            {{ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES }}));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    auto cnnNetwork = testEnv->network;
    auto exeNetwork = testEnv->exeNetwork;

    if (param.device == CommonTestUtils::DEVICE_CPU || param.device == CommonTestUtils::DEVICE_GPU) {
        CNNNetwork execGraph;
        ASSERT_NO_THROW(execGraph = exeNetwork.GetExecGraphInfo());

        // Store all the original layers from the network
        const std::vector<CNNLayerPtr> originalLayers = CNNNetSortTopologically(cnnNetwork);
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            originalLayersMap[layer->name] = 0;
        }

        // Store all the layers from the executable graph information represented as CNNNetwork
        const std::vector<CNNLayerPtr> execGraphLayers = CNNNetSortTopologically(execGraph);
        bool has_layer_with_valid_time = false;
        for (const auto &execLayer : execGraphLayers) {
            // At least one layer in the topology should be executed and have valid perf counter value
            try {
                float x = static_cast<float>(std::atof(execLayer->params[ExecGraphInfoSerialization::PERF_COUNTER].c_str()));
                ASSERT_GE(x, 0.0f);
                has_layer_with_valid_time = true;
            } catch (std::exception&) { }

            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = execLayer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES];
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');

            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            } );
        }

        ASSERT_TRUE(has_layer_with_valid_time);

        // All layers from the original IR must be present within ExecGraphInfo
        for (auto& layer : originalLayersMap) {
            ASSERT_GT(layer.second, 0);
        }
    } else {
        // Not implemented for other plugins
        ASSERT_THROW(exeNetwork.GetExecGraphInfo(), InferenceEngineException);
    }
}

TEST_P(BehaviorPluginTestExecGraphInfo, CheckExecGraphInfoSerialization) {
    auto param = GetParam();

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));

    auto cnnNetwork = testEnv->network;
    auto exeNetwork = testEnv->exeNetwork;

    if (param.device == CommonTestUtils::DEVICE_CPU || param.device == CommonTestUtils::DEVICE_GPU) {
        CNNNetwork execGraph;
        ASSERT_NO_THROW(execGraph = exeNetwork.GetExecGraphInfo());
        execGraph.serialize("exeNetwork.xml", "exeNetwork.bin");
    } else {
        // Not implemented for other plugins
        ASSERT_THROW(exeNetwork.GetExecGraphInfo(), InferenceEngineException);
    }
}
