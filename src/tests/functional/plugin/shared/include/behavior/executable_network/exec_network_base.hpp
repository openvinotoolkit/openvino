// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exec_graph_info.hpp>
#include "base/behavior_test_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/core/model.hpp"

namespace BehaviorTestsDefinitions {
class ExecutableNetworkBaseTest : public testing::WithParamInterface<BehaviorTestsUtils::InferRequestParams>,
                                  public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BehaviorTestsUtils::InferRequestParams> obj) {
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace CommonTestUtils;
            result << "config=" << configuration;
        }
        return result.str();
    }

    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(targetDevice, configuration) = this->GetParam();
        ie = PluginCache::get().ie(targetDevice);
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
    }

protected:
    InferenceEngine::CNNNetwork cnnNet;
    std::shared_ptr<InferenceEngine::Core> ie;
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutable) {
    ASSERT_NO_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableWithIncorrectConfig) {
    std::map<std::string, std::string> incorrectConfig = {{ "abc", "def" }};
    ASSERT_ANY_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, incorrectConfig));
}

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCreateInferRequest) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(auto req = execNet.CreateInferRequest());
}

TEST_P(ExecutableNetworkBaseTest, checkGetExecGraphInfoIsNotNullptr) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    InferenceEngine::CNNNetwork execGraph = execNet.GetExecGraphInfo();
    ASSERT_NE(execGraph.getFunction(), nullptr);
}

TEST_P(ExecutableNetworkBaseTest, checkGetMetric) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(execNet.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
}

TEST_P(ExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCheckConfig) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    for (const auto& configItem : configuration) {
        InferenceEngine::Parameter param;
        ASSERT_NO_THROW(param = execNet.GetConfig(configItem.first));
        ASSERT_FALSE(param.empty());
        ASSERT_EQ(param, InferenceEngine::Parameter(configItem.second));
    }
}

TEST_P(ExecutableNetworkBaseTest, canSetConfigToExecNet) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto& confItem : configuration) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    ASSERT_NO_THROW(execNet.SetConfig(config));
}

TEST_P(ExecutableNetworkBaseTest, canSetConfigToExecNetWithIncorrectConfig) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    std::map<std::string, std::string> incorrectConfig = {{ "abc", "def" }};
    std::map<std::string, InferenceEngine::Parameter> config;
    for (const auto& confItem : incorrectConfig) {
        config.insert({confItem.first, InferenceEngine::Parameter(confItem.second)});
    }
    ASSERT_ANY_THROW(execNet.SetConfig(config));
}

TEST_P(ExecutableNetworkBaseTest, canSetConfigToExecNetAndCheckConfigAndCheck) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
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
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(auto inInfo = execNet.GetInputsInfo());
}

TEST_P(ExecutableNetworkBaseTest,  CanGetOutputsInfo) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(auto outInfo = execNet.GetOutputsInfo());
}

TEST_P(ExecutableNetworkBaseTest,  CanGetInputsInfoAndCheck) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto inInfo = execNet.GetInputsInfo();
    auto inCnnInfo = cnnNet.getInputsInfo();
    for (const auto& itemInInfo : inCnnInfo) {
        ASSERT_NE(inInfo.find(itemInInfo.first), inInfo.end());
    }
}

TEST_P(ExecutableNetworkBaseTest,  CanGetOutputsInfoAndCheck) {
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto outInfo = execNet.GetOutputsInfo();
    auto outCnnInfo = cnnNet.getOutputsInfo();
    for (const auto& itemOutInfo : outCnnInfo) {
        ASSERT_NE(outInfo.find(itemOutInfo.first), outInfo.end());
    }
}

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoBeforeExecution) {
    InferenceEngine::CNNNetwork execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
    std::map<std::string, int> originalLayersMap;
    for (const auto &layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;

    auto getFunction = execGraph.getFunction();
    ASSERT_NE(getFunction, nullptr);

    for (const auto & op : getFunction->get_ops()) {
        const ov::RTMap& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
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

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoAfterExecution) {
    InferenceEngine::CNNNetwork execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
    std::map<std::string, int> originalLayersMap;
    for (const auto &layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;
    // Store all the layers from the executable graph information represented as CNNNetwork
    bool hasOpWithValidTime = false;
    auto getFunction = execGraph.getFunction();
    ASSERT_NE(nullptr, getFunction);

    for (const auto & op : getFunction->get_ops()) {
        const auto & rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
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

TEST_P(ExecutableNetworkBaseTest, CheckExecGraphInfoSerialization) {
    auto ts = CommonTestUtils::GetTimestamp();
    std::string out_xml_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".xml";
    std::string out_bin_path = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts + ".bin";

    InferenceEngine::CNNNetwork execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(execGraph = execNet.GetExecGraphInfo());
    ASSERT_NO_THROW(execGraph.serialize(out_xml_path, out_bin_path));
    CommonTestUtils::removeIRFiles(out_xml_path, out_bin_path);
}

TEST_P(ExecutableNetworkBaseTest, canExport) {
    auto ts = CommonTestUtils::GetTimestamp();
    std::string modelName = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts;
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    ASSERT_NO_THROW(execNet.Export(modelName));
    ASSERT_TRUE(CommonTestUtils::fileExists(modelName + ".xml"));
    ASSERT_TRUE(CommonTestUtils::fileExists(modelName + ".bin"));
    CommonTestUtils::removeIRFiles(modelName + ".xml", modelName + ".bin");
}

TEST_P(ExecutableNetworkBaseTest, pluginDoesNotChangeOriginalNetwork) {
    // compare 2 networks
    auto referenceNetwork = ngraph::builder::subgraph::makeConvPoolRelu();
    compare_functions(cnnNet.getFunction(), referenceNetwork);
}

using ExecNetSetPrecision = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(ExecNetSetPrecision, canSetInputPrecisionForNetwork) {
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::InputsDataMap inputs_info = cnnNet.getInputsInfo();
    ASSERT_EQ(1u, inputs_info.size());
    inputs_info.begin()->second->setPrecision(netPrecision);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(ExecNetSetPrecision, canSetOutputPrecisionForNetwork) {
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::OutputsDataMap outputs_info = cnnNet.getOutputsInfo();
    ASSERT_EQ(outputs_info.size(), 1u);
    outputs_info.begin()->second->setPrecision(netPrecision);
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}
TEST_P(ExecutableNetworkBaseTest, loadIncorrectV10Model) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("data1");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result = std::make_shared<ov::opset8::Result>(relu);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1});
        function->get_rt_info()["version"] = int64_t(10);
        function->set_friendly_name("SimpleReLU");
    }
    InferenceEngine::CNNNetwork cnnNet(function);
    EXPECT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

TEST_P(ExecutableNetworkBaseTest, loadIncorrectV11Model) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("data1");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result = std::make_shared<ov::opset8::Result>(relu);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1});
        function->get_rt_info()["version"] = int64_t(11);
        function->set_friendly_name("SimpleReLU");
    }
    InferenceEngine::CNNNetwork cnnNet(function);
    EXPECT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
}

}  // namespace BehaviorTestsDefinitions
