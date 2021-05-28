// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/variant.hpp>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include <exec_graph_info.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include <chrono>

namespace BehaviorTestsDefinitions {
using ExecGraphTests = BehaviorTestsUtils::BehaviorTestsBasic;

inline std::vector<std::string> separateStrToVec(std::string str, const char sep) {
    std::vector<std::string> result;

    std::istringstream stream(str);
    std::string strVal;

    while (getline(stream, strVal, sep)) {
        result.push_back(strVal);
    }
    return result;
}

namespace {
    std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        auto epoch = now.time_since_epoch();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
        return std::to_string(ns.count());
    }

    std::string test_name() {
        std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::replace_if(test_name.begin(), test_name.end(),
                        [](char c) { return (c == '/' || c == '='); }, '_');
        return test_name;
    }
} // namespace

TEST_P(ExecGraphTests, CheckExecGraphInfoBeforeExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice != CommonTestUtils::DEVICE_MULTI &&
        targetDevice != CommonTestUtils::DEVICE_TEMPLATE &&
        targetDevice != CommonTestUtils::DEVICE_GNA) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        // Store all the original layers from the network
        const auto originalLayers = function->get_ops();
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            originalLayersMap[layer->get_friendly_name()] = 0;
        }
        int IteratorForLayersConstant = 0;

        auto function = execGraph.getFunction();
        ASSERT_NE(function, nullptr);

        for (const auto & op : function->get_ops()) {
            const auto & rtInfo = op->get_rt_info();

            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                IE_ASSERT(nullptr != value);

                return value->get();
            };

            // Each layer from the execGraphInfo network must have PM data option set
            ASSERT_EQ("not_executed", getExecValue(ExecGraphInfoSerialization::PERF_COUNTER));
            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
            if (origFromExecLayer == "")
                IteratorForLayersConstant++;
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }

        // All layers from the original IR must be present with in ExecGraphInfo
        for (auto &layer : originalLayersMap) {
            if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
                IteratorForLayersConstant--;
                continue;
            }
            ASSERT_GE(layer.second, 0);
        }
    } else {
        InferenceEngine::ExecutableNetwork network;
        ASSERT_NO_THROW(network = ie->LoadNetwork(cnnNet, targetDevice, configuration));
        ASSERT_THROW(network.GetExecGraphInfo(), InferenceEngine::NotImplemented);
    }
}

TEST_P(ExecGraphTests, CheckExecGraphInfoAfterExecution) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice != CommonTestUtils::DEVICE_MULTI &&
        targetDevice != CommonTestUtils::DEVICE_TEMPLATE &&
        targetDevice != CommonTestUtils::DEVICE_GNA) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        // Store all the original layers from the network
        const auto originalLayers = function->get_ops();
        std::map<std::string, int> originalLayersMap;
        for (const auto &layer : originalLayers) {
            originalLayersMap[layer->get_friendly_name()] = 0;
        }
        int IteratorForLayersConstant = 0;
        // Store all the layers from the executable graph information represented as CNNNetwork
        bool has_layer_with_valid_time = false;
        auto function = execGraph.getFunction();
        ASSERT_NE(nullptr, function);

        for (const auto & op : function->get_ops()) {
            const auto & rtInfo = op->get_rt_info();

            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                IE_ASSERT(nullptr != value);

                return value->get();
            };

            // At least one layer in the topology should be executed and have valid perf counter value
            try {
                float x = static_cast<float>(std::atof(
                        getExecValue(ExecGraphInfoSerialization::PERF_COUNTER).c_str()));
                ASSERT_GE(x, 0.0f);
                has_layer_with_valid_time = true;
            } catch (std::exception &) {}

            // Parse origin layer names (fused/merged layers) from the executable graph
            // and compare with layers from the original model
            auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
            std::vector<std::string> origFromExecLayerSep = separateStrToVec(origFromExecLayer, ',');
            if (origFromExecLayer == "")
                IteratorForLayersConstant++;
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }

        ASSERT_TRUE(has_layer_with_valid_time);

        // All layers from the original IR must be present within ExecGraphInfo
        for (auto &layer : originalLayersMap) {
            if ((layer.second == 0) && (IteratorForLayersConstant > 0)) {
                IteratorForLayersConstant--;
                continue;
            }
            ASSERT_GE(layer.second, 0);
        }
    } else {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration).GetExecGraphInfo(),
                InferenceEngine::NotImplemented);
    }
}

TEST_P(ExecGraphTests, CheckExecGraphInfoSerialization) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string out_xml_path = test_name() + "_" + timestamp() + ".xml";
    std::string out_bin_path = test_name() + "_" + timestamp() + ".bin";

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice != CommonTestUtils::DEVICE_MULTI &&
        targetDevice != CommonTestUtils::DEVICE_TEMPLATE &&
        targetDevice != CommonTestUtils::DEVICE_GNA) {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        execGraph.serialize(out_xml_path, out_bin_path);
        ASSERT_EQ(0, std::remove(out_xml_path.c_str()));
        ASSERT_EQ(0, std::remove(out_bin_path.c_str()));
    } else {
        ASSERT_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration).GetExecGraphInfo(),
                     InferenceEngine::NotImplemented);
    }
}
}  // namespace BehaviorTestsDefinitions