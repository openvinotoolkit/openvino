// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <fstream>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/concat_with_params.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/single_split.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::test::behavior {
namespace {
using op::v0::Parameter, op::v0::Constant, op::v1::Add;

Tensor from_stream(std::istream& stream, const size_t size) {
    ov::Tensor t(element::from<char>(), Shape{size});
    stream.read(t.data<char>(), size);
    return t;
}

std::shared_ptr<Model> make_model_with_weights(const ov::Tensor& w) {
    constexpr auto precision = element::f32;

    auto weights = std::make_shared<Constant>(w);
    auto input = std::make_shared<Parameter>(precision, Shape{1});
    auto add = std::make_shared<Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
    util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
    return model;
}

std::shared_ptr<Model> make_model_with_weights() {
    constexpr auto precision = element::f32;

    auto weights = std::make_shared<Constant>(element::f32, Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<Parameter>(precision, Shape{1});
    auto add = std::make_shared<Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
    util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
    return model;
}
}  // namespace

using testing::_;

namespace utils = ::ov::test::utils;

class OVCompiledModelBaseTest : public testing::WithParamInterface<InferRequestParams>,
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
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;

    void set_api_entity() override {
        api_entity = ov::test::utils::ov_entity::ov_compiled_model;
    }
};

using OVAutoExecutableNetworkTest = OVCompiledModelBaseTest;
using OVCompiledModelBaseTestOptional = OVCompiledModelBaseTest;

TEST_P(OVCompiledModelBaseTest, canCompileModel) {
    EXPECT_NO_THROW(auto execNet = core->compile_model(function, target_device, configuration));
}

TEST_P(OVCompiledModelBaseTest, canCompileModelFromMemory) {
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

TEST_P(OVCompiledModelBaseTest, canCompileModelwithBrace) {
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
    ov::CompiledModel compiled_model;
    {
        ov::Core tmp_core = ov::test::utils::create_core();
        compiled_model = tmp_core.compile_model(model, ov::Tensor(), target_device, configuration);
    }
    EXPECT_NO_THROW(compiled_model.get_property(ov::optimal_number_of_infer_requests));
}

TEST(OVCompiledModelBaseTest, canCompileModelToDefaultDevice) {
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function = ov::test::utils::make_single_concat_with_constant();
    EXPECT_NO_THROW(auto execNet = core->compile_model(function));
}

TEST_P(OVCompiledModelBaseTest, canCompileModelAndCreateInferRequest) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto req = execNet.create_infer_request());
}

TEST_P(OVCompiledModelBaseTestOptional, checkGetExecGraphInfoIsNotNullptr) {
    auto execNet = core->compile_model(function, target_device, configuration);
    auto execGraph = execNet.get_runtime_model();
    EXPECT_NE(execGraph, nullptr);
}

TEST_P(OVCompiledModelBaseTest, canCreateTwoCompiledModel) {
    std::vector<ov::CompiledModel> vec;
    for (auto i = 0; i < 2; i++) {
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, target_device, configuration)));
        EXPECT_NE(nullptr, function);
    }
}

TEST_P(OVCompiledModelBaseTest, CanGetInputsInfo) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto inInfo = execNet.inputs());
}

TEST_P(OVCompiledModelBaseTest, CanCreateTwoCompiledModelsAndCheckRuntimeModel) {
    std::vector<ov::CompiledModel> vec;
    for (size_t i = 0; i < 2; i++) {
        EXPECT_NO_THROW(vec.push_back(core->compile_model(function, target_device, configuration)));
        EXPECT_NE(nullptr, vec[i].get_runtime_model());
        EXPECT_NE(vec.begin()->get_runtime_model(), vec[i].get_runtime_model());
    }
}

TEST_P(OVCompiledModelBaseTest, pluginDoesNotChangeOriginalNetwork) {
    // compare 2 networks
    auto referenceNetwork = ov::test::utils::make_conv_pool_relu();
    compare_functions(function, referenceNetwork);
}

TEST_P(OVCompiledModelBaseTest, CanSetInputPrecisionForNetwork) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_single_concat_with_constant();
    ov::Core core = ov::test::utils::create_core();
    auto ppp = ov::preprocess::PrePostProcessor(model);
    ov::preprocess::InputInfo& input = ppp.input();
    input.model().set_layout("??HW");
    input.preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    model = ppp.build();
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration));
}

TEST_P(OVCompiledModelBaseTest, CanSetOutputPrecisionForNetwork) {
    std::shared_ptr<ov::Model> model = ov::test::utils::make_single_concat_with_constant();
    ov::Core core = ov::test::utils::create_core();
    auto ppp = ov::preprocess::PrePostProcessor(model);
    ov::preprocess::OutputInfo& output = ppp.output();
    output.postprocess().convert_element_type(ov::element::u8);
    model = ppp.build();
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration));
}

TEST_P(OVCompiledModelBaseTest, CanGetOutputsInfo) {
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(auto outInfo = execNet.outputs());
}

TEST_P(OVCompiledModelBaseTest, CanGetInputsInfoAndCheck) {
    auto execNet = core->compile_model(function, target_device, configuration);
    auto inputs = execNet.inputs();
    std::vector<std::string> paramVec;
    for (const auto& input : inputs) {
        paramVec.push_back(input.get_tensor().get_any_name());
    }
    auto params = function->get_parameters();
    for (const auto& param : params) {
        EXPECT_NE(std::find(paramVec.begin(), paramVec.end(), param->get_output_tensor(0).get_any_name()),
                  paramVec.end());
    }
}

TEST_P(OVCompiledModelBaseTest, CanGetOutputsInfoAndCheck) {
    auto execNet = core->compile_model(function, target_device, configuration);
    auto outputs = execNet.outputs();
    std::vector<std::string> resVec;
    for (const auto& out : outputs) {
        resVec.push_back(out.get_tensor().get_any_name());
    }
    auto results = function->get_results();
    for (const auto& param : results) {
        EXPECT_NE(std::find(resVec.begin(), resVec.end(), param->get_output_tensor(0).get_any_name()), resVec.end());
    }
}

TEST_P(OVCompiledModelBaseTestOptional, CheckExecGraphInfoBeforeExecution) {
    std::shared_ptr<const ov::Model> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = core->compile_model(function, target_device, configuration);
    EXPECT_NO_THROW(execGraph = execNet.get_runtime_model());
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;

    std::shared_ptr<const ov::Model> getFunction = ov::as_type_ptr<const ov::Model>(execGraph);
    ASSERT_NE(getFunction, nullptr);

    for (const auto& op : getFunction->get_ops()) {
        const ov::RTMap& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        // Each layer from the execGraphInfo network must have PM data option set
        EXPECT_EQ("not_executed", getExecValue(ov::exec_model_info::PERF_COUNTER));
        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ov::exec_model_info::ORIGINAL_NAMES);
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

TEST_P(OVCompiledModelBaseTestOptional, CheckExecGraphInfoAfterExecution) {
    std::shared_ptr<const ov::Model> execGraph;
    // Load CNNNetwork to target plugins
    auto execNet = core->compile_model(function, target_device, configuration);
    execNet.create_infer_request().infer();
    EXPECT_NO_THROW(execGraph = execNet.get_runtime_model());
    std::map<std::string, int> originalLayersMap;
    for (const auto& layer : function->get_ops()) {
        originalLayersMap[layer->get_friendly_name()] = 0;
    }
    int constCnt = 0;
    // Store all the layers from the executable graph information represented as CNNNetwork
    bool hasOpWithValidTime = false;
    auto getFunction = ov::as_type_ptr<const ov::Model>(execGraph);
    ASSERT_NE(nullptr, getFunction);

    for (const auto& op : getFunction->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        // At least one layer in the topology should be executed and have valid perf counter value
        try {
            float x = static_cast<float>(std::atof(getExecValue(ov::exec_model_info::PERF_COUNTER).c_str()));
            std::cout << "TIME: " << x << std::endl;
            EXPECT_GE(x, 0.0f);
            hasOpWithValidTime = true;
        } catch (std::exception&) {
        }

        // Parse origin layer names (fused/merged layers) from the executable graph
        // and compare with layers from the original model
        auto origFromExecLayer = getExecValue(ov::exec_model_info::ORIGINAL_NAMES);
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

TEST_P(OVCompiledModelBaseTest, CheckExecGraphInfoSerialization) {
    auto filePrefix = ov::test::utils::generateTestFilePrefix();
    std::string out_xml_path = filePrefix + ".xml";
    std::string out_bin_path = filePrefix + ".bin";

    std::shared_ptr<const ov::Model> runtime_model;

    auto compiled_model = core->compile_model(function, target_device, configuration);
    OV_ASSERT_NO_THROW(runtime_model = compiled_model.get_runtime_model());
    OV_ASSERT_NO_THROW(ov::serialize(runtime_model, out_xml_path, out_bin_path));
    ov::test::utils::removeIRFiles(out_xml_path, out_bin_path);
}

TEST_P(OVCompiledModelBaseTest, getInputFromFunctionWithSingleInput) {
    ov::CompiledModel execNet;

    function = ov::test::utils::make_split_concat();

    execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->inputs().size(), execNet.inputs().size());
    EXPECT_NO_THROW(execNet.input());
    EXPECT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
    EXPECT_EQ(function->input().get_tensor().get_partial_shape(), execNet.input().get_tensor().get_partial_shape());
    EXPECT_EQ(function->input().get_tensor().get_element_type(), execNet.input().get_tensor().get_element_type());
}

TEST_P(OVCompiledModelBaseTest, getOutputFromFunctionWithSingleInput) {
    ov::CompiledModel execNet;

    function = ov::test::utils::make_split_concat();

    execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_NO_THROW(execNet.output());
    EXPECT_EQ(function->output().get_tensor().get_names(), execNet.output().get_tensor().get_names());
    EXPECT_EQ(function->output().get_tensor().get_partial_shape(), execNet.output().get_tensor().get_partial_shape());
    EXPECT_EQ(function->output().get_tensor().get_element_type(), execNet.output().get_tensor().get_element_type());
}

TEST_P(OVCompiledModelBaseTest, getInputsFromFunctionWithSeveralInputs) {
    ov::CompiledModel execNet;

    function = ov::test::utils::make_concat_with_params();

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

TEST_P(OVCompiledModelBaseTest, getOutputsFromFunctionWithSeveralOutputs) {
    ov::CompiledModel execNet;

    function = ov::test::utils::make_multiple_input_output_double_concat();

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
    EXPECT_EQ(function->output(1).get_node(), function->output("concat2").get_node());
    EXPECT_NE(function->output(0).get_node(), function->output("concat2").get_node());
    EXPECT_EQ(function->output(0).get_node(), function->output("concat1").get_node());
    EXPECT_NE(function->output(1).get_node(), function->output("concat1").get_node());
}

TEST_P(OVCompiledModelBaseTest, getOutputsFromSplitFunctionWithSeveralOutputs) {
    ov::CompiledModel execNet;

    function = ov::test::utils::make_single_split();

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
TEST_P(OVCompiledModelBaseTest, precisionsAsInOriginalFunction) {
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

TEST_P(OVCompiledModelBaseTest, loadIncorrectV10Model) {
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{param1, param2}, 1);
        concat->set_friendly_name("data1");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});
        function->get_rt_info()["version"] = int64_t(10);
        function->set_friendly_name("SimpleConcat");
    }
    EXPECT_THROW(core->compile_model(function, target_device, configuration), ov::Exception);
}

TEST_P(OVCompiledModelBaseTest, loadIncorrectV11Model) {
    ov::CompiledModel execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, ov::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{param1, param2}, 1);
        concat->set_friendly_name("data1");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});
        function->get_rt_info()["version"] = int64_t(11);
        function->set_friendly_name("SimpleConcat");
    }
    EXPECT_NO_THROW(core->compile_model(function, target_device, configuration));
}

TEST_P(OVCompiledModelBaseTest, canLoadCorrectNetworkToGetExecutableWithIncorrectConfig) {
    std::map<std::string, ov::Any> config = {{"abc", "def"}};
    for (const auto& confItem : configuration) {
        config.emplace(confItem.first, confItem.second);
    }
    bool is_meta_devices = target_device.find("AUTO") != std::string::npos ||
                           target_device.find("MULTI") != std::string::npos ||
                           target_device.find("HETERO") != std::string::npos;
    if (is_meta_devices) {
        EXPECT_NO_THROW(auto execNet = core->compile_model(function, target_device, config));
    } else {
        EXPECT_ANY_THROW(auto execNet = core->compile_model(function, target_device, config));
    }
}

typedef std::tuple<ov::element::Type,  // Type to convert
                   std::string,        // Device name
                   ov::AnyMap          // Config
                   >
    CompiledModelSetTypeParams;

class CompiledModelSetType : public testing::WithParamInterface<CompiledModelSetTypeParams>,
                             public OVCompiledNetworkTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompiledModelSetTypeParams> obj) {
        ov::element::Type convert_type;
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(convert_type, target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');

        std::ostringstream result;
        result << "ConvertType=" << convert_type.get_type_name() << "_";
        result << "targetDevice=" << target_device << "_";
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
        return result.str();
    }

    void SetUp() override {
        std::tie(convert_type, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }
    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

    ov::element::Type convert_type;
    ov::AnyMap configuration;
};

TEST_P(CompiledModelSetType, canSetInputTypeAndCompileModel) {
    auto model = ov::test::utils::make_conv_pool_relu();

    ov::Core core = ov::test::utils::create_core();
    auto ppp = ov::preprocess::PrePostProcessor(model);
    auto& input = ppp.input();
    input.preprocess().convert_element_type(convert_type);
    model = ppp.build();
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration));
}

TEST_P(CompiledModelSetType, canSetOutputTypeAndCompileModel) {
    auto model = ov::test::utils::make_conv_pool_relu();

    ov::Core core = ov::test::utils::create_core();
    auto ppp = ov::preprocess::PrePostProcessor(model);
    auto& output = ppp.output();
    output.postprocess().convert_element_type(convert_type);
    model = ppp.build();
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration));
}

TEST_P(CompiledModelSetType, canSetInputOutputTypeAndCompileModel) {
    auto model = ov::test::utils::make_conv_pool_relu();

    ov::Core core = ov::test::utils::create_core();
    auto ppp = ov::preprocess::PrePostProcessor(model);
    auto& input = ppp.input();
    input.preprocess().convert_element_type(convert_type);
    auto& output = ppp.output();
    output.postprocess().convert_element_type(convert_type);
    model = ppp.build();
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, configuration));
}

TEST_P(OVCompiledModelBaseTest, import_from_weightless_blob) {
    const auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});

    std::stringstream export_stream;
    {
        configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        auto model = make_model_with_weights();
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        compiled_model.export_model(export_stream);
        if (target_device != utils::DEVICE_HETERO) {
            EXPECT_EQ(ov::CacheMode::OPTIMIZE_SIZE, compiled_model.get_property(ov::cache_mode));
        }
    }
    OV_EXPECT_THROW(core->import_model(export_stream, target_device), ov::Exception, _);

    utils::removeFile(w_file_path.string());
}

TEST_P(OVCompiledModelBaseTest, compile_from_regular_blob) {
    auto compiled_model_ref = core->compile_model(make_model_with_weights(), target_device, configuration);
    ASSERT_TRUE(compiled_model_ref);
    if (target_device != utils::DEVICE_HETERO) {
        EXPECT_EQ(ov::CacheMode::OPTIMIZE_SPEED, compiled_model_ref.get_property(ov::cache_mode));
    }
    Tensor exported_model;
    {
        std::stringstream export_stream;
        compiled_model_ref.export_model(export_stream);
        exported_model = from_stream(export_stream, export_stream.str().size());
    }

    ov::Tensor expected, output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer reference model
    {
        auto infer_request = compiled_model_ref.create_infer_request();
        infer_request.set_tensor("input", input);
        infer_request.infer();
        expected = infer_request.get_tensor("add");
    }
    // Infer compiled from stream
    {
        configuration.emplace(ov::hint::compiled_blob(exported_model));
        auto empty_model = std::make_shared<Model>(OutputVector{}, ParameterVector{}, "Empty model");
        auto model_from_stream = core->compile_model(empty_model, target_device, configuration);
        auto infer_request_import = model_from_stream.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    OV_ASSERT_NO_THROW(utils::compare(expected, output));
}

TEST_P(OVCompiledModelBaseTest, compile_from_weightless_blob) {
    auto weights = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});

    Tensor exported_model;
    configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    {
        auto model = make_model_with_weights(weights);
        std::stringstream export_stream;
        auto export_cfg = configuration;
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model_ref = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model_ref);
        compiled_model_ref.export_model(export_stream);
        exported_model = from_stream(export_stream, export_stream.str().size());
    }

    auto expected = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{3.0f, 5.0f, 3.0f, 3.0f, 3.0f});
    {
        // store weights in file, not same as in original model model
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    ov::Tensor output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer compiled from weightless stream
    {
        configuration.emplace(ov::weights_path(w_file_path.string()));
        configuration.emplace(ov::hint::compiled_blob(exported_model));
        auto empty_model = std::make_shared<Model>(OutputVector{}, ParameterVector{}, "Empty model");
        auto import_model = core->compile_model(empty_model, target_device, configuration);
        auto infer_request_import = import_model.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    utils::removeFile(w_file_path.string());
    OV_ASSERT_NO_THROW(utils::compare(expected, output));
}

TEST_P(OVCompiledModelBaseTest, compile_from_weightless_blob_but_no_weights) {
    Tensor exported_model;
    auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});

    auto model = make_model_with_weights();
    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model_ref = core->compile_model(model, target_device, export_cfg);
        std::stringstream export_stream;
        ASSERT_TRUE(compiled_model_ref);
        compiled_model_ref.export_model(export_stream);
        exported_model = from_stream(export_stream, export_stream.str().size());
        if (target_device != utils::DEVICE_HETERO) {
            EXPECT_EQ(ov::CacheMode::OPTIMIZE_SIZE, compiled_model_ref.get_property(ov::cache_mode));
        }
    }

    auto expected = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f, 3.0f});
    ov::Tensor output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer compiled from weightless stream and no weights use original model
    {
        configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        configuration.emplace(ov::hint::compiled_blob(exported_model));
        configuration.emplace(ov::weights_path(w_file_path.string()));
        auto import_model = core->compile_model(model, target_device, configuration);
        auto infer_request_import = import_model.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    utils::removeFile(w_file_path.string());
    OV_ASSERT_NO_THROW(utils::compare(expected, output));
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_use_weight_hint) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, "weights.bin"});
    {
        // store weights in file, not same as in original model model
        std::filesystem::create_directories(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    configuration.emplace(ov::weights_path(w_file_path.string()));
    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();

    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }
    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_no_hint) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, "weights.bin"});
    {
        // store weights in file, not same as in original model model
        std::filesystem::create_directories(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();

    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }
    {
        configuration.emplace(ov::weights_path(w_file_path.string()));
        if (target_device == utils::DEVICE_GPU) {
            configuration.emplace(ov::hint::model(model));
            configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, use_blob_hint_has_priority_over_cache) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, "weights.bin"});
    auto blob_file_path = cache_dir / "export.blob";

    {
        // store weights in file, not same as in original model model
        std::filesystem::create_directories(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();

    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::weights_path(w_file_path.string()));
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
        std::ofstream blob_file(blob_file_path, std::ios::binary);
        compiled_model.export_model(blob_file);
    }
    {
        configuration.emplace(ov::hint::compiled_blob(ov::read_tensor_data(blob_file_path)));
        if (target_device == utils::DEVICE_GPU) {
            configuration.emplace(ov::hint::model(model));
            configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, use_blob_hint_has_priority_over_cache_but_weights_bind_from_model_hint) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, "weights.bin"});
    auto blob_file_path = cache_dir / "export.blob";
    std::filesystem::create_directories(cache_dir);

    {
        // store weights in file, not same as in original model model
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();
    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
        auto blob_file = std::ofstream(blob_file_path, std::ios::binary);
        compiled_model.export_model(blob_file);
    }
    {
        model->get_rt_info()["__weights_path"] = w_file_path.string();
        configuration.emplace(ov::hint::compiled_blob(ov::read_tensor_data(blob_file_path)));
        if (target_device == utils::DEVICE_GPU) {
            configuration.emplace(ov::hint::model(model));
            configuration.emplace(ov::weights_path(w_file_path.string()));
            configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, use_blob_hint_has_priority_over_cache_but_weights_from_model_path) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto model_file_path = cache_dir / "model.xml";
    auto w_file_path = model_file_path;
    w_file_path.replace_extension(".bin");
    auto blob_file_path = cache_dir / "export.blob";
    std::filesystem::create_directories(cache_dir);

    auto model = make_model_with_weights();
    ov::save_model(model, model_file_path, false);

    configuration.emplace(ov::cache_dir(cache_dir.string()));
    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
        auto blob_file = std::ofstream(blob_file_path, std::ios::binary);
        compiled_model.export_model(blob_file);
    }
    {
        configuration.emplace(ov::hint::compiled_blob(ov::read_tensor_data(blob_file_path)));
        if (target_device == utils::DEVICE_GPU) {
            configuration.emplace(ov::hint::model(model));
            configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto compiled_model = core->compile_model(model_file_path, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, use_blob_hint_which_fails_load_from_cache) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, "weights.bin"});

    {
        // store weights in file, not same as in original model model
        std::filesystem::create_directories(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();

    {
        auto export_cfg = configuration;
        export_cfg.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        if (target_device == utils::DEVICE_GPU) {
            export_cfg.emplace(ov::hint::model(model));
            export_cfg.emplace(ov::weights_path(w_file_path.string()));
        }
        auto compiled_model = core->compile_model(model, target_device, export_cfg);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }
    {
        configuration.emplace(ov::weights_path(w_file_path.string()));
        configuration.emplace(ov::hint::compiled_blob(Tensor{}));
        if (target_device == utils::DEVICE_GPU) {
            configuration.emplace(ov::hint::model(model));
            configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_but_no_weights) {
    auto cache_dir = ov::util::Path(utils::getCurrentWorkingDir()) / (utils::generateTestFilePrefix() + "cache");
    auto w_file_path = cache_dir / "weights.bin";
    std::filesystem::create_directories(cache_dir);

    configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    configuration.emplace(ov::cache_dir(cache_dir.string()));
    auto model = make_model_with_weights();

    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }
    {
        // model not loaded from cache as no weights on path
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_TRUE(compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));
    }

    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
}
}  // namespace ov::test::behavior
