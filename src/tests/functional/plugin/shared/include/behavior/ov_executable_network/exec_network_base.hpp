// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <exec_graph_info.hpp>
#include <fstream>
#include <transformations/serialize.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVExecutableNetworkBaseTest : public testing::WithParamInterface<InferRequestParams>,
                                    public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
    }

    bool compareTensors(const ov::Tensor& t1, const ov::Tensor& t2) {
        void* data1;
        void* data2;
        try {
            data1 = t1.data();
        } catch (const ov::Exception&) {
            // Remote tensor
            data1 = nullptr;
        }
        try {
            data2 = t2.data();
        } catch (const ov::Exception&) {
            // Remote tensor
            data2 = nullptr;
        }
        return t1.get_element_type() == t2.get_element_type() && t1.get_shape() == t2.get_shape() &&
               t1.get_byte_size() == t2.get_byte_size() && t1.get_size() == t2.get_size() &&
               t1.get_strides() == t2.get_strides() && data1 == data2;
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
};

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutable) {
    EXPECT_NO_THROW(auto execNet = core->compile_model(function, targetDevice, configuration));
}

TEST(OVExecutableNetworkBaseTest, smoke_LoadNetworkToDefaultDeviceNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function = ngraph::builder::subgraph::makeConvPoolRelu();
    EXPECT_NO_THROW(auto execNet = core->compile_model(function));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableWithIncorrectConfig) {
    ov::AnyMap incorrectConfig = {{"abc", "def"}};
    EXPECT_ANY_THROW(auto execNet = core->compile_model(function, targetDevice, incorrectConfig));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCreateInferRequest) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(auto req = execNet.create_infer_request());
}

TEST_P(OVExecutableNetworkBaseTest, checkGetExecGraphInfoIsNotNullptr) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    auto execGraph = execNet.get_runtime_model();
    EXPECT_NE(execGraph, nullptr);
}

TEST_P(OVExecutableNetworkBaseTest, checkGetMetric) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(execNet.get_property(ov::supported_properties));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCheckConfig) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    for (const auto& configItem : configuration) {
        ov::Any param;
        EXPECT_NO_THROW(param = execNet.get_property(configItem.first));
        EXPECT_FALSE(param.empty());
        EXPECT_EQ(param, configItem.second);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNet) {
    auto execNet = core->compile_model(function, targetDevice);
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : configuration) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_NO_THROW(execNet.set_property(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetWithIncorrectConfig) {
    auto execNet = core->compile_model(function, targetDevice);
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : incorrectConfig) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_ANY_THROW(execNet.set_property(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetAndCheckConfigAndCheck) {
    auto execNet = core->compile_model(function, targetDevice);
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : configuration) {
        config.emplace(confItem.first, confItem.second);
    }
    execNet.set_property(config);
    for (const auto& configItem : configuration) {
        ov::Any param;
        EXPECT_NO_THROW(param = execNet.get_property(configItem.first));
        EXPECT_FALSE(param.empty());
        EXPECT_EQ(param, configItem.second);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanCreateTwoExeNetworks) {
    std::vector<ov::CompiledModel> vec;
    for (auto i = 0; i < 2; i++) {
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, targetDevice, configuration)));
        EXPECT_NE(nullptr, function);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanCreateTwoExeNetworksAndCheckFunction) {
    std::vector<ov::CompiledModel> vec;
    for (auto i = 0; i < 2; i++) {
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, targetDevice, configuration)));
        EXPECT_NE(nullptr, vec[i].get_runtime_model());
        EXPECT_NE(vec.begin()->get_runtime_model(), vec[i].get_runtime_model());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfo) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(auto inInfo = execNet.inputs());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetOutputsInfo) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(auto outInfo = execNet.outputs());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfoAndCheck) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    auto inputs = execNet.inputs();
    std::vector<std::string> paramVec;
    for (const auto& input : inputs) {
        paramVec.push_back(*input.get_tensor().get_names().begin());
    }
    auto params = function->get_parameters();
    for (const auto& param : params) {
        EXPECT_NE(std::find(paramVec.begin(), paramVec.end(), *param->get_output_tensor(0).get_names().begin()),
                  paramVec.end());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanGetOutputsInfoAndCheck) {
    auto execNet = core->compile_model(function, targetDevice, configuration);
    auto outputs = execNet.outputs();
    std::vector<std::string> resVec;
    for (const auto& out : outputs) {
        resVec.push_back(*out.get_tensor().get_names().begin());
    }
    auto results = function->get_results();
    for (const auto& param : results) {
        EXPECT_NE(std::find(resVec.begin(), resVec.end(), *param->get_output_tensor(0).get_names().begin()),
                  resVec.end());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CheckExecGraphInfoBeforeExecution) {
    std::shared_ptr<const ov::Model> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(execGraph = execNet.get_runtime_model());
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;

    std::shared_ptr<const ngraph::Function> getFunction = std::dynamic_pointer_cast<const ngraph::Function>(execGraph);
    EXPECT_NE(getFunction, nullptr);

    for (const auto& op : getFunction->get_ops()) {
        const ov::RTMap& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        // Each layer from the execGraphInfo network must have PM data option set
        EXPECT_EQ("not_executed", getExecValue(ExecGraphInfoSerialization::PERF_COUNTER));
        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
        if (origFromExecLayer.empty()) {
            constCnt++;
        } else {
            auto origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string& op) {
                auto origLayer = originalLayersMap.find(op);
                EXPECT_NE(originalLayersMap.end(), origLayer) << op;
                origLayer->second++;
            });
        }
    }

    // All layers from the original IR must be present with in ExecGraphInfo
    for (auto& layer : originalLayersMap) {
        if ((layer.second == 0) && (constCnt > 0)) {
            constCnt--;
        } else {
            EXPECT_GE(layer.second, 0);
        }
    }
}

TEST_P(OVExecutableNetworkBaseTest, CheckExecGraphInfoAfterExecution) {
    std::shared_ptr<const ov::Model> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_NO_THROW(execGraph = execNet.get_runtime_model());
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;
    // Store all the layers from the executable graph information represented as CNNNetwork
    bool hasOpWithValidTime = false;
    auto getFunction = std::dynamic_pointer_cast<const ngraph::Function>(execGraph);
    EXPECT_NE(nullptr, getFunction);

    for (const auto& op : getFunction->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        // At least one layer in the topology should be executed and have valid perf counter value
        try {
            float x = static_cast<float>(std::atof(getExecValue(ExecGraphInfoSerialization::PERF_COUNTER).c_str()));
            std::cout << "TIME: " << x << std::endl;
            EXPECT_GE(x, 0.0f);
            hasOpWithValidTime = true;
        } catch (std::exception&) {
        }

        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ExecGraphInfoSerialization::ORIGINAL_NAMES);
        std::vector<std::string> origFromExecLayerSep = CommonTestUtils::splitStringByDelimiter(origFromExecLayer);
        if (origFromExecLayer.empty()) {
            constCnt++;
        } else {
            std::for_each(origFromExecLayerSep.begin(), origFromExecLayerSep.end(), [&](const std::string& layer) {
                auto origLayer = originalLayersMap.find(layer);
                EXPECT_NE(originalLayersMap.end(), origLayer) << layer;
                origLayer->second++;
            });
        }
    }

    EXPECT_TRUE(hasOpWithValidTime);

    // All layers from the original IR must be present within ExecGraphInfo
    for (auto& layer : originalLayersMap) {
        if ((layer.second == 0) && (constCnt > 0)) {
            constCnt--;
        } else {
            EXPECT_GE(layer.second, 0);
        }
    }
}

TEST_P(OVExecutableNetworkBaseTest, canExport) {
    auto ts = CommonTestUtils::GetTimestamp();
    std::string modelName = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts;
    auto execNet = core->compile_model(function, targetDevice, configuration);
    std::ofstream out(modelName, std::ios::out);
    EXPECT_NO_THROW(execNet.export_model(out));
    out.close();
    EXPECT_TRUE(CommonTestUtils::fileExists(modelName + ".xml"));
    EXPECT_TRUE(CommonTestUtils::fileExists(modelName + ".bin"));
    CommonTestUtils::removeIRFiles(modelName + ".xml", modelName + ".bin");
}

TEST_P(OVExecutableNetworkBaseTest, pluginDoesNotChangeOriginalNetwork) {
    // compare 2 networks
    auto referenceNetwork = ngraph::builder::subgraph::makeConvPoolRelu();
    compare_functions(function, referenceNetwork);
}

TEST_P(OVExecutableNetworkBaseTest, getInputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->inputs().size(), execNet.inputs().size());
    EXPECT_NO_THROW(execNet.input());
    EXPECT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
    EXPECT_EQ(function->input().get_tensor().get_partial_shape(), execNet.input().get_tensor().get_partial_shape());
    EXPECT_EQ(function->input().get_tensor().get_element_type(), execNet.input().get_tensor().get_element_type());

    ov::InferRequest request = execNet.create_infer_request();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.input()));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.input().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVExecutableNetworkBaseTest, getOutputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_NO_THROW(execNet.output());
    EXPECT_EQ(function->output().get_tensor().get_names(), execNet.output().get_tensor().get_names());
    EXPECT_EQ(function->output().get_tensor().get_partial_shape(), execNet.output().get_tensor().get_partial_shape());
    EXPECT_EQ(function->output().get_tensor().get_element_type(), execNet.output().get_tensor().get_element_type());

    ov::InferRequest request = execNet.create_infer_request();
    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.output()));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.output().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVExecutableNetworkBaseTest, getInputsFromFunctionWithSeveralInputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("relu_op");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result1 = std::make_shared<ov::opset8::Result>(relu);
        result1->set_friendly_name("result1");
        auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
        concat->set_friendly_name("concat_op");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result2 = std::make_shared<ov::opset8::Result>(concat);
        result2->set_friendly_name("result2");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{param1, param2});
        function->set_friendly_name("SimpleReLU");
    }
    execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), execNet.inputs().size());
    EXPECT_THROW(execNet.input(), ov::Exception);
    EXPECT_EQ(function->input(0).get_tensor().get_names(), execNet.input(0).get_tensor().get_names());
    EXPECT_EQ(function->input(0).get_tensor().get_partial_shape(), execNet.input(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(0).get_tensor().get_element_type(), execNet.input(0).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_tensor().get_names(), execNet.input(1).get_tensor().get_names());
    EXPECT_EQ(function->input(1).get_tensor().get_partial_shape(), execNet.input(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(1).get_tensor().get_element_type(), execNet.input(1).get_tensor().get_element_type());
    EXPECT_EQ(function->input(0).get_node(), function->input("data1").get_node());
    EXPECT_NE(function->input(1).get_node(), function->input("data1").get_node());
    EXPECT_EQ(function->input(1).get_node(), function->input("data2").get_node());
    EXPECT_NE(function->input(0).get_node(), function->input("data2").get_node());

    ov::InferRequest request = execNet.create_infer_request();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.input(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.input(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.input(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.input(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->input(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVExecutableNetworkBaseTest, getOutputsFromFunctionWithSeveralOutputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("relu_op");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result1 = std::make_shared<ov::opset8::Result>(relu);
        result1->set_friendly_name("result1");
        auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
        concat->set_friendly_name("concat_op");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result2 = std::make_shared<ov::opset8::Result>(concat);
        result2->set_friendly_name("result2");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{param1, param2});
        function->set_friendly_name("SimpleReLU");
    }
    execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_THROW(execNet.output(), ov::Exception);
    EXPECT_EQ(function->output(0).get_tensor().get_names(), execNet.output(0).get_tensor().get_names());
    EXPECT_EQ(function->output(0).get_tensor().get_partial_shape(), execNet.output(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(0).get_tensor().get_element_type(), execNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_tensor().get_names(), execNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(), execNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(), execNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(function->output(0).get_node(), function->output("relu").get_node());
    EXPECT_NE(function->output(1).get_node(), function->output("relu").get_node());
    EXPECT_EQ(function->output(1).get_node(), function->output("concat").get_node());
    EXPECT_NE(function->output(0).get_node(), function->output("concat").get_node());

    ov::InferRequest request = execNet.create_infer_request();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.output(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.output(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVExecutableNetworkBaseTest, getOutputsFromSplitFunctionWithSeveralOutputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ngraph::Shape({1, 4, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto axis_node = ov::opset8::Constant::create(element::i64, Shape{}, {1});
        auto split = std::make_shared<ov::opset8::Split>(param1, axis_node, 2);
        split->set_friendly_name("split");
        split->output(0).get_tensor().set_names({"tensor_split_1"});
        split->output(1).get_tensor().set_names({"tensor_split_2"});
        auto result1 = std::make_shared<ov::opset8::Result>(split->output(0));
        result1->set_friendly_name("result1");
        auto result2 = std::make_shared<ov::opset8::Result>(split->output(1));
        result2->set_friendly_name("result2");
        function =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2}, ngraph::ParameterVector{param1});
        function->set_friendly_name("SingleSplit");
    }
    execNet = core->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_THROW(execNet.output(), ov::Exception);
    EXPECT_EQ(function->output(0).get_tensor().get_names(), execNet.output(0).get_tensor().get_names());
    EXPECT_EQ(function->output(0).get_tensor().get_partial_shape(), execNet.output(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(0).get_tensor().get_element_type(), execNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_tensor().get_names(), execNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(), execNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(), execNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(function->output(0).get_node(), function->output("tensor_split_1").get_node());
    EXPECT_NE(function->output(1).get_node(), function->output("tensor_split_1").get_node());
    EXPECT_EQ(function->output(1).get_node(), function->output("tensor_split_2").get_node());
    EXPECT_NE(function->output(0).get_node(), function->output("tensor_split_2").get_node());

    ov::InferRequest request = execNet.create_infer_request();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.output(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(execNet.output(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(execNet.output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(function->output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

// Load correct network to Plugin to get executable network
TEST_P(OVExecutableNetworkBaseTest, precisionsAsInOriginalFunction) {
    ov::CompiledModel execNet;
    EXPECT_NO_THROW(execNet = core->compile_model(function, targetDevice, configuration));

    EXPECT_EQ(function->get_parameters().size(), execNet.inputs().size());
    auto ref_parameter = function->get_parameters().back();
    auto actual_parameter = execNet.inputs().back().get_node_shared_ptr();
    EXPECT_EQ(ref_parameter->get_element_type(), actual_parameter->get_element_type());
    EXPECT_EQ(ref_parameter->get_shape(), actual_parameter->get_shape());
    EXPECT_EQ(ref_parameter->get_friendly_name(), actual_parameter->get_friendly_name());

    EXPECT_EQ(function->get_results().size(), execNet.outputs().size());
    auto ref_result = function->get_results().back();
    auto actual_result = execNet.outputs().back().get_node_shared_ptr();
    EXPECT_EQ(ref_result->get_element_type(), actual_result->get_element_type());
    EXPECT_EQ(ref_result->get_shape(), actual_result->get_shape());
    EXPECT_EQ(ref_result->get_friendly_name(), actual_result->get_friendly_name());
}

// Load correct network to Plugin to get executable network
TEST_P(OVExecutableNetworkBaseTest, precisionsAsInOriginalIR) {
    const std::string m_out_xml_path_1 = "precisionsAsInOriginalIR.xml";
    const std::string m_out_bin_path_1 = "precisionsAsInOriginalIR.bin";
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(function);

    ov::CompiledModel execNet;
    EXPECT_NO_THROW(execNet = core->compile_model(m_out_xml_path_1, targetDevice, configuration));
    CommonTestUtils::removeIRFiles(m_out_xml_path_1, m_out_bin_path_1);

    EXPECT_EQ(function->get_parameters().size(), execNet.inputs().size());
    auto ref_parameter = function->get_parameters().back();
    auto actual_parameter = execNet.inputs().back().get_node_shared_ptr();
    EXPECT_EQ(ref_parameter->get_element_type(), actual_parameter->get_element_type());
    EXPECT_EQ(ref_parameter->get_shape(), actual_parameter->get_shape());
    EXPECT_EQ(ref_parameter->get_friendly_name(), actual_parameter->get_friendly_name());

    EXPECT_EQ(function->get_results().size(), execNet.outputs().size());
    auto ref_result = function->get_results().back();
    auto actual_result = execNet.outputs().back().get_node_shared_ptr();
    EXPECT_EQ(ref_result->get_element_type(), actual_result->get_element_type());
    EXPECT_EQ(ref_result->get_shape(), actual_result->get_shape());
    EXPECT_EQ(ref_result->get_friendly_name(), actual_result->get_friendly_name());
}

TEST_P(OVExecutableNetworkBaseTest, getCompiledModelFromInferRequest) {
    ov::InferRequest req;
    {
        ov::CompiledModel compiled_model;
        ASSERT_NO_THROW(compiled_model = core->compile_model(function, targetDevice, configuration));
        ASSERT_NO_THROW(req = compiled_model.create_infer_request());
        ASSERT_NO_THROW(req.infer());
    }
    {
        ov::CompiledModel restored_compiled_model;
        ov::InferRequest another_req;
        ASSERT_NO_THROW(restored_compiled_model = req.get_compiled_model());
        ASSERT_NO_THROW(another_req = restored_compiled_model.create_infer_request());
        ASSERT_NO_THROW(another_req.infer());
    }
}

TEST_P(OVExecutableNetworkBaseTest, loadIncorrectV10Model) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("data1");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result = std::make_shared<ov::opset8::Result>(relu);
        result->set_friendly_name("result");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        function->get_rt_info()["version"] = int64_t(10);
        function->set_friendly_name("SimpleReLU");
    }
    EXPECT_THROW(core->compile_model(function, targetDevice, configuration), ov::Exception);
}

TEST_P(OVExecutableNetworkBaseTest, loadIncorrectV11Model) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("data1");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result = std::make_shared<ov::opset8::Result>(relu);
        result->set_friendly_name("result");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        function->get_rt_info()["version"] = int64_t(11);
        function->set_friendly_name("SimpleReLU");
    }
    EXPECT_NO_THROW(core->compile_model(function, targetDevice, configuration));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
