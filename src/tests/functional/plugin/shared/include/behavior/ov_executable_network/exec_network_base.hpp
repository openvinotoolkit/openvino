// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <exec_graph_info.hpp>
#include <fstream>
#include <openvino/pass/serialize.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace behavior {

// ===================== DEPRECATED =====================

class OVExecutableNetworkBaseTest : public testing::WithParamInterface<InferRequestParams>,
                                    public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

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
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;

    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ov_compiled_model; }
};

using OVAutoExecutableNetworkTest = OVExecutableNetworkBaseTest;

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutable) {
    EXPECT_NO_THROW(auto execNet = core->compile_model(function, target_device, configuration));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadNetworkFromMemory) {
    std::string model = R"V0G0N(
        <net name="Network" version="10">
            <layers>
                <layer name="in1" type="Parameter" id="0" version="opset8">
                    <data element_type="f16" shape="1,3,22,22"/>
                    <output>
                        <port id="0" precision="FP16" names="data1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </output>
                </layer>
                <layer name="in2" type="Parameter" id="1" version="opset8">
                    <data element_type="f16" shape="1,3,22,22"/>
                    <output>
                        <port id="0" precision="FP16" names="data2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </output>
                </layer>
                <layer name="concat" id="2" type="Concat" version="opset8">
                    <input>
                        <port id="0" precision="FP16">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                        <port id="1"  precision="FP16">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP16" names="r">
                            <dim>1</dim>
                            <dim>6</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </output>
                </layer>
                <layer name="output" type="Result" id="3" version="opset8">
                    <input>
                        <port id="0" precision="FP16">
                            <dim>1</dim>
                            <dim>6</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            </edges>
        </net>
        )V0G0N";
    EXPECT_NO_THROW(auto execNet = core->compile_model(model, ov::Tensor(), target_device, configuration));
}

TEST(OVExecutableNetworkBaseTest, smoke_LoadNetworkToDefaultDeviceNoThrow) {
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function = ngraph::builder::subgraph::makeSingleConcatWithConstant();
    EXPECT_NO_THROW(auto execNet = core->compile_model(function));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCreateInferRequest) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto req = execNet.create_infer_request());
}

TEST_P(OVExecutableNetworkBaseTest, checkGetExecGraphInfoIsNotNullptr) {
    auto execNet = core->compile_model(function, target_device, configuration);
    auto execGraph = execNet.get_runtime_model();
    EXPECT_NE(execGraph, nullptr);
}

TEST_P(OVExecutableNetworkBaseTest, checkGetMetric) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(execNet.get_property(ov::supported_properties));
}

TEST_P(OVExecutableNetworkBaseTest, canLoadCorrectNetworkToGetExecutableAndCheckConfig) {
    auto execNet = core->compile_model(function, target_device, configuration);
    for (const auto& configItem : configuration) {
        ov::Any param;
        EXPECT_NO_THROW(param = execNet.get_property(configItem.first));
        EXPECT_FALSE(param.empty());
        EXPECT_EQ(param, configItem.second);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNet) {
    auto execNet = core->compile_model(function, target_device);
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : configuration) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_NO_THROW(execNet.set_property(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetWithIncorrectConfig) {
    auto execNet = core->compile_model(function, target_device);
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : incorrectConfig) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_ANY_THROW(execNet.set_property(config));
}

TEST_P(OVExecutableNetworkBaseTest, CanSetConfigToExecNetAndCheckConfigAndCheck) {
    auto execNet = core->compile_model(function, target_device);
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
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, target_device, configuration)));
        EXPECT_NE(nullptr, function);
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanCreateTwoExeNetworksAndCheckFunction) {
    std::vector<ov::CompiledModel> vec;
    for (auto i = 0; i < 2; i++) {
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, target_device, configuration)));
        EXPECT_NE(nullptr, vec[i].get_runtime_model());
        EXPECT_NE(vec.begin()->get_runtime_model(), vec[i].get_runtime_model());
    }
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfo) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto inInfo = execNet.inputs());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetOutputsInfo) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto outInfo = execNet.outputs());
}

TEST_P(OVExecutableNetworkBaseTest, CanGetInputsInfoAndCheck) {
    auto execNet = core->compile_model(function, target_device, configuration);
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
    auto execNet = core->compile_model(function, target_device, configuration);
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
    auto execNet = core->compile_model(function, target_device, configuration);
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
            auto origFromExecLayerSep = ov::test::utils::splitStringByDelimiter(origFromExecLayer);
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
    auto execNet = core->compile_model(function, target_device, configuration);
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
        std::vector<std::string> origFromExecLayerSep = ov::test::utils::splitStringByDelimiter(origFromExecLayer);
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

TEST_P(OVExecutableNetworkBaseTest, LoadNetworkCreateDefaultExecGraphResult) {
    auto net = core->compile_model(function, target_device, configuration);
    auto runtime_function = net.get_runtime_model();
    ASSERT_NE(nullptr, runtime_function);
    auto actual_parameters = runtime_function->get_parameters();
    auto actual_results = runtime_function->get_results();
    auto expected_parameters = function->get_parameters();
    auto expected_results = function->get_results();
    ASSERT_EQ(expected_parameters.size(), actual_parameters.size());
    for (std::size_t i = 0; i < expected_parameters.size(); ++i) {
        auto expected_element_type = expected_parameters[i]->get_output_element_type(0);
        auto actual_element_type = actual_parameters[i]->get_output_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_parameters[i]->get_output_shape(0);
        auto actual_shape = actual_parameters[i]->get_output_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
    ASSERT_EQ(expected_results.size(), actual_results.size());
    for (std::size_t i = 0; i < expected_results.size(); ++i) {
        auto expected_element_type = expected_results[i]->get_input_element_type(0);
        auto actual_element_type = actual_results[i]->get_input_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_results[i]->get_input_shape(0);
        auto actual_shape = actual_results[i]->get_input_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
}

TEST_P(OVExecutableNetworkBaseTest, canExport) {
    auto ts = ov::test::utils::GetTimestamp();
    std::string modelName = GetTestName().substr(0, ov::test::utils::maxFileNameLength) + "_" + ts;
    auto execNet = core->compile_model(function, target_device, configuration);
    std::ofstream out(modelName, std::ios::out);
    EXPECT_NO_THROW(execNet.export_model(out));
    out.close();
    EXPECT_TRUE(ov::test::utils::fileExists(modelName));
    ov::test::utils::removeFile(modelName);
}

TEST_P(OVExecutableNetworkBaseTest, pluginDoesNotChangeOriginalNetwork) {
    // compare 2 networks
    auto referenceNetwork = ngraph::builder::subgraph::makeConvPoolRelu();
    compare_functions(function, referenceNetwork);
}

TEST_P(OVExecutableNetworkBaseTest, getInputFromFunctionWithSingleInput) {
    ov::CompiledModel execNet;

    function = ngraph::builder::subgraph::makeSplitConcat();

    execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->inputs().size(), execNet.inputs().size());
    EXPECT_NO_THROW(execNet.input());
    EXPECT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
    EXPECT_EQ(function->input().get_tensor().get_partial_shape(), execNet.input().get_tensor().get_partial_shape());
    EXPECT_EQ(function->input().get_tensor().get_element_type(), execNet.input().get_tensor().get_element_type());
}

TEST_P(OVExecutableNetworkBaseTest, getOutputFromFunctionWithSingleInput) {
    ov::CompiledModel execNet;

    function = ngraph::builder::subgraph::makeSplitConcat();

    execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_NO_THROW(execNet.output());
    EXPECT_EQ(function->output().get_tensor().get_names(), execNet.output().get_tensor().get_names());
    EXPECT_EQ(function->output().get_tensor().get_partial_shape(), execNet.output().get_tensor().get_partial_shape());
    EXPECT_EQ(function->output().get_tensor().get_element_type(), execNet.output().get_tensor().get_element_type());
}

TEST_P(OVExecutableNetworkBaseTest, getInputsFromFunctionWithSeveralInputs) {
    ov::CompiledModel execNet;

    function = ngraph::builder::subgraph::makeConcatWithParams();

    execNet = core->compile_model(function, target_device, configuration);
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
}

TEST_P(OVExecutableNetworkBaseTest, getOutputsFromFunctionWithSeveralOutputs) {
    ov::CompiledModel execNet;

    function = ngraph::builder::subgraph::makeMultipleInputOutputDoubleConcat();

    execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_THROW(execNet.output(), ov::Exception);
    EXPECT_EQ(function->output(0).get_tensor().get_names(), execNet.output(0).get_tensor().get_names());
    EXPECT_EQ(function->output(0).get_tensor().get_partial_shape(), execNet.output(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(0).get_tensor().get_element_type(), execNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_tensor().get_names(), execNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(), execNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(), execNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(function->output(0).get_node(), function->output("concat1").get_node());
    EXPECT_NE(function->output(0).get_node(), function->output("concat2").get_node());
    EXPECT_EQ(function->output(1).get_node(), function->output("concat2").get_node());
    EXPECT_NE(function->output(1).get_node(), function->output("concat1").get_node());
}

TEST_P(OVExecutableNetworkBaseTest, getOutputsFromSplitFunctionWithSeveralOutputs) {
    ov::CompiledModel execNet;

    function = ngraph::builder::subgraph::makeSingleSplit();

    execNet = core->compile_model(function, target_device, configuration);
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
}

// Load correct network to Plugin to get executable network
TEST_P(OVExecutableNetworkBaseTest, precisionsAsInOriginalFunction) {
    ov::CompiledModel execNet;
    EXPECT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

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
    auto filePrefix = ov::test::utils::generateTestFilePrefix();
    const std::string m_out_xml_path_1 = filePrefix + "precisionsAsInOriginalIR.xml";
    const std::string m_out_bin_path_1 = filePrefix + "precisionsAsInOriginalIR.bin";
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(function);

    ov::CompiledModel execNet;
    EXPECT_NO_THROW(execNet = core->compile_model(m_out_xml_path_1, target_device, configuration));
    ov::test::utils::removeIRFiles(m_out_xml_path_1, m_out_bin_path_1);

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
        ASSERT_NO_THROW(compiled_model = core->compile_model(function, target_device, configuration));
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
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{param1, param2}, 1);
        concat->set_friendly_name("data1");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        result->set_friendly_name("result");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});
        function->get_rt_info()["version"] = int64_t(10);
        function->set_friendly_name("SimpleConcat");
    }
    EXPECT_THROW(core->compile_model(function, target_device, configuration), ov::Exception);
}

TEST_P(OVExecutableNetworkBaseTest, loadIncorrectV11Model) {
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ngraph::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{param1, param2}, 1);
        concat->set_friendly_name("data1");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        result->set_friendly_name("result");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});
        function->get_rt_info()["version"] = int64_t(11);
        function->set_friendly_name("SimpleConcat");
    }
    EXPECT_NO_THROW(core->compile_model(function, target_device, configuration));
}

TEST_P(OVAutoExecutableNetworkTest, AutoNotImplementedSetConfigToExecNet) {
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : configuration) {
        config.emplace(confItem.first, confItem.second);
    }
    auto execNet = core->compile_model(function, target_device, config);
    EXPECT_ANY_THROW(execNet.set_property(config));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
