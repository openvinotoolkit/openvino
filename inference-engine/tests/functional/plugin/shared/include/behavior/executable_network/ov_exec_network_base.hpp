// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <fstream>

#include <exec_graph_info.hpp>
#include "base/behavior_test_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVExecutableNetworkBaseTest : public BehaviorTestsUtils::OVInferRequestTests {
    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutable) {
    ASSERT_NO_THROW(execNet = core->compile_model(function, targetDevice, configuration));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableWithIncorrectConfig) {
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    ASSERT_ANY_THROW(execNet = core->compile_model(function, targetDevice, configuration));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCreateInferRequest) {
    execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(auto req = execNet.create_infer_request());
}

TEST_P(OVExecutableNetworkBaseTest, checkGetExecGraphInfoIsNotNullptr) {
    execNet = core->compile_model(function, targetDevice, configuration);
    auto execGraph = execNet.get_runtime_function();
    ASSERT_NE(execGraph, nullptr);
}

TEST_P(OVExecutableNetworkBaseTest, checkGetMetric) {
    execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execNet.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCheckConfig) {
    execNet = core->compile_model(function, targetDevice, configuration);
    for (const auto &configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = execNet.get_config(configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNet) {
    execNet = core->compile_model(function, targetDevice);
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto &confItem : configuration) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    ASSERT_NO_THROW(execNet.set_config(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetWithIncorrectConfig) {
    execNet = core->compile_model(function, targetDevice);
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto &confItem : incorrectConfig) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    ASSERT_ANY_THROW(execNet.set_config(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetAndCheckConfigAndCheck) {
    execNet = core->compile_model(function, targetDevice);
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto &confItem : configuration) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    execNet.set_config(config);
    for (const auto &configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = execNet.get_config(configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanCreateTwoExeNetworks) {
    std::vector<ov::runtime::ExecutableNetwork> vec;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(vec.push_back(core->compile_model(function, targetDevice, configuration)));
        ASSERT_NE(nullptr, function);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanCreateTwoExeNetworksAndCheckFunction) {
    std::vector<ov::runtime::ExecutableNetwork> vec;
    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(vec.push_back(core->compile_model(function, targetDevice, configuration)));
        ASSERT_NE(nullptr, vec[i].get_runtime_function());
        ASSERT_NE(vec.begin()->get_runtime_function(), vec[i].get_runtime_function());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfo) {
    execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(auto inInfo = execNet.get_parameters());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetOutputsInfo) {
    execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(auto outInfo = execNet.get_results());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfoAndCheck) {
    execNet = core->compile_model(function, targetDevice, configuration);
    auto inInfo = execNet.get_parameters();
    auto inCnnInfo = function->get_parameters();
    for (const auto &itemInInfo : inCnnInfo) {
        ASSERT_NE(std::find(inInfo.begin(), inInfo.end(), itemInInfo), inInfo.end());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanGetOutputsInfoAndCheck) {
    execNet = core->compile_model(function, targetDevice, configuration);
    auto outInfo = execNet.get_results();
    auto outCnnInfo = function->get_results();
    for (const auto &itemOutInfo : outCnnInfo) {
        ASSERT_NE(std::find(outCnnInfo.begin(), outCnnInfo.end(), itemOutInfo), outCnnInfo.end());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CheckExecGraphInfoBeforeExecution) {
    std::shared_ptr<const ov::Function> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.get_runtime_function());
    std::map<std::string, int> originalLayersMap;
    for (const auto &layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;

    std::shared_ptr<const ngraph::Function> getFunction = std::dynamic_pointer_cast<const ngraph::Function>(execGraph);
    ASSERT_NE(getFunction, nullptr);

    for (const auto &op : getFunction->get_ops()) {
        const ov::RTMap &rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string &paramName) -> std::string {
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
        if (origFromExecLayer.empty()) {
            constCnt++;
        } else {
            auto origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &op) {
                auto origLayer = originalLayersMap.find(op);
                ASSERT_NE(originalLayersMap.end(), origLayer) << op;
                origLayer->second++;
            });
        }
    }

    // All layers from the original IR must be present with in ExecGraphInfo
    for (auto &layer : originalLayersMap) {
        if ((layer.second == 0) && (constCnt > 0)) {
            constCnt--;
        } else {
            ASSERT_GE(layer.second, 0);
        }
    }
}

TEST_P(OVExecutableNetworkBaseTest, CheckExecGraphInfoAfterExecution) {
    std::shared_ptr<const ov::Function> execGraph;
    // Load CNNNetwork to target plugins
    execNet = core->compile_model(function, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.get_runtime_function());
    std::map<std::string, int> originalLayersMap;
    for (const auto &layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;
    // Store all the layers from the executable graph information represented as CNNNetwork
    bool hasOpWithValidTime = false;
    auto getFunction = std::dynamic_pointer_cast<const ngraph::Function>(execGraph);
    ASSERT_NE(nullptr, getFunction);

    for (const auto &op : getFunction->get_ops()) {
        const auto &rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string &paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        // At least one layer in the topology should be executed and have valid perf counter value
        try {
            float x = static_cast<float>(std::atof(getExecValue(ExecGraphInfoSerialization::PERF_COUNTER).c_str()));
            std::cout << "TIME: " << x << std::endl;
            ASSERT_GE(x, 0.0f);
            hasOpWithValidTime = true;
        } catch (std::exception &) {}

        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
        std::vector<std::string> origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
        if (origFromExecLayer.empty()) {
            constCnt++;
        } else {
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string &layer) {
                auto origLayer = originalLayersMap.find(layer);
                ASSERT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }
    }

    ASSERT_TRUE(hasOpWithValidTime);

    // All layers from the original IR must be present within ExecGraphInfo
    for (auto &layer : originalLayersMap) {
        if ((layer.second == 0) && (constCnt > 0)) {
            constCnt--;
        } else {
            ASSERT_GE(layer.second, 0);
        }
    }
}

TEST_P(OVExecutableNetworkBaseTest, canExport) {
    auto ts = CommonTestUtils::GetTimestamp();
    std::string modelName = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts;
    auto execNet = core->compile_model(function, targetDevice, configuration);
    std::ofstream out(modelName, std::ios::out);
    ASSERT_NO_THROW(execNet.export_model(out));
    out.close();
    ASSERT_TRUE(CommonTestUtils::fileExists(modelName + ".xml"));
    ASSERT_TRUE(CommonTestUtils::fileExists(modelName + ".bin"));
    CommonTestUtils::removeIRFiles(modelName + ".xml", modelName + ".bin");
}

TEST_P(OVExecutableNetworkBaseTest, pluginDoesNotChangeOriginalNetwork) {
    // compare 2 networks
    auto referenceNetwork = ngraph::builder::subgraph::makeConvPoolRelu();
    compare_functions(referenceNetwork, function);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov