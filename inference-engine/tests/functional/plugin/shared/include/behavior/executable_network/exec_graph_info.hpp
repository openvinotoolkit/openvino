// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exec_graph_info.hpp>
#include "base/behavior_test_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace BehaviorTestsDefinitions {

using ExecutableNetworkBaseTest = BehaviorTestsUtils::InferRequestTests;

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutable) {
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(ExecutableNetworkBaseTest, checkGetExecGraphInfoIsNotNullptr) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    InferenceEngine::CNNNetwork execGraph = execNet.GetExecGraphInfo();
    ASSERT_NE(execGraph.getFunction(), nullptr);
}

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCheckConfig) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = execNet.GetConfig(configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(ExecutableNetworkBaseTest, canSetConfigToExecNet) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice);
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto& confItem : configuration) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    ASSERT_NO_THROW(execNet.SetConfig(config));
}

TEST_P(ExecutableNetworkBaseTest, canSetConfigToExecNetAndCheckConfigAndCheck) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice);
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto& confItem : configuration) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    execNet.SetConfig(config);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = execNet.GetConfig(configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(ExecutableNetworkBaseTest,  CanCreateTwoExeNetworks) {
    std::vector<InferenceEngine::ExecutableNetwork> vec;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(vec.push_back(ie->LoadNetwork(cnnNet, targetDevice, configuration)));
        ASSERT_NE(nullptr, cnnNet.getFunction());
    }
}

TEST_P(ExecutableNetworkBaseTest,  CanCreateTwoExeNetworksAndCheckFunction) {
    std::vector<InferenceEngine::ExecutableNetwork> vec;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(vec.push_back(ie->LoadNetwork(cnnNet, targetDevice, configuration)));
        ASSERT_NE(nullptr, vec[i].GetExecGraphInfo().getFunction());
        ASSERT_NE(vec.begin()->GetExecGraphInfo().getFunction(), vec[i].GetExecGraphInfo().getFunction());
    }
}

TEST_P(ExecutableNetworkBaseTest,  CanGetInputsInfo) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(auto inInfo = execNet.GetInputsInfo());
}

TEST_P(ExecutableNetworkBaseTest,  CanGetOutputsInfo) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(auto outInfo = execNet.GetOutputsInfo());
}

TEST_P(ExecutableNetworkBaseTest,  CanGetInputsInfoAndCheck) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto inInfo = execNet.GetInputsInfo();
    auto inCnnInfo = cnnNet.getInputsInfo();
    for (const auto& itemInInfo : inCnnInfo) {
        ASSERT_NE(inInfo.find(itemInInfo.first), inInfo.end());
    }
}

TEST_P(ExecutableNetworkBaseTest,  CanGetOutputsInfoAndCheck) {
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto outInfo = execNet.GetOutputsInfo();
    auto outCnnInfo = cnnNet.getOutputsInfo();
    for (const auto& itemOutInfo : outCnnInfo) {
        ASSERT_NE(outInfo.find(itemOutInfo.first), outInfo.end());
    }
}

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoBeforeExecution) {
    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
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

        auto getFunction = execGraph.getFunction();
        ASSERT_NE(getFunction, nullptr);

        for (const auto & op : getFunction->get_ops()) {
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
            std::vector<std::string> origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
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

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoAfterExecution) {
    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
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
        auto getFunction = execGraph.getFunction();
        ASSERT_NE(nullptr, getFunction);

        for (const auto & op : getFunction->get_ops()) {
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
            std::vector<std::string> origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
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

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoSerialization) {
    auto ts = CommonTestUtils::GetTimestamp();
    std::string out_xml_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".xml";
    std::string out_bin_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".bin";

    InferenceEngine::CNNNetwork execGraph;
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
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