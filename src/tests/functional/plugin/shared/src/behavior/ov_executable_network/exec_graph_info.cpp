// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include "behavior/ov_executable_network/exec_graph_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVExecGraphImportExportTest::getTestCaseName(testing::TestParamInfo<OVExecGraphImportExportTestParams> obj) {
    ov::element::Type_t elementType;
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(elementType, targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    result << "elementType=" << elementType << "_";
    if (!configuration.empty()) {
        result << "config=(";
        for (const auto& config : configuration) {
            result << config.first << "=";
            config.second.print(result);
            result << "_";
        }
        result << ")";
    }
    return result.str();
}

void OVExecGraphImportExportTest::SetUp() {
    std::tie(elementType, target_device, configuration) = this->GetParam();
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    APIBaseTest::SetUp();
}

void OVExecGraphImportExportTest::TearDown() {
    if (!configuration.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

TEST_P(OVExecGraphImportExportTest, importExportedFunction) {
    if (target_device == ov::test::utils::DEVICE_MULTI || target_device == ov::test::utils::DEVICE_AUTO) {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    ov::CompiledModel execNet;
    // Create simple function
    function = ov::test::utils::make_multiple_input_output_double_concat({1, 2, 24, 24}, elementType);
    execNet = core->compile_model(function, target_device, configuration);

    std::stringstream strm;
    execNet.export_model(strm);

    ov::CompiledModel importedExecNet = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), importedExecNet.inputs().size());
    EXPECT_THROW(importedExecNet.input(), ov::Exception);
    EXPECT_EQ(function->input(0).get_tensor().get_names(), importedExecNet.input(0).get_tensor().get_names());
    EXPECT_EQ(function->input(0).get_tensor().get_partial_shape(),
              importedExecNet.input(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(0).get_tensor().get_element_type(),
              importedExecNet.input(0).get_tensor().get_element_type());
    EXPECT_EQ(function->input(0).get_element_type(),
              importedExecNet.input(0).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_tensor().get_names(), importedExecNet.input(1).get_tensor().get_names());
    EXPECT_EQ(function->input(1).get_tensor().get_partial_shape(),
              importedExecNet.input(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(1).get_tensor().get_element_type(),
              importedExecNet.input(1).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_element_type(),
              importedExecNet.input(1).get_tensor().get_element_type());
    EXPECT_EQ(importedExecNet.input(0).get_node(), importedExecNet.input("data1").get_node());
    EXPECT_NE(importedExecNet.input(1).get_node(), importedExecNet.input("data1").get_node());
    EXPECT_EQ(importedExecNet.input(1).get_node(), importedExecNet.input("data2").get_node());
    EXPECT_NE(importedExecNet.input(0).get_node(), importedExecNet.input("data2").get_node());
    EXPECT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), importedExecNet.outputs().size());
    EXPECT_THROW(importedExecNet.output(), ov::Exception);
    EXPECT_EQ(function->output(0).get_tensor().get_names(), importedExecNet.output(0).get_tensor().get_names());
    EXPECT_EQ(function->output(0).get_tensor().get_partial_shape(),
              importedExecNet.output(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(0).get_tensor().get_element_type(),
              importedExecNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(0).get_element_type(),
              importedExecNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_tensor().get_names(), importedExecNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(),
              importedExecNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(),
              importedExecNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_element_type(),
              importedExecNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(importedExecNet.output(0).get_node(), importedExecNet.output("concat1").get_node());
    EXPECT_NE(importedExecNet.output(1).get_node(), importedExecNet.output("concat1").get_node());
    EXPECT_EQ(importedExecNet.output(1).get_node(), importedExecNet.output("concat2").get_node());
    EXPECT_NE(importedExecNet.output(0).get_node(), importedExecNet.output("concat2").get_node());
    EXPECT_THROW(importedExecNet.input("param1"), ov::Exception);
    EXPECT_THROW(importedExecNet.input("param2"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("result1"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("result2"), ov::Exception);
}

TEST_P(OVExecGraphImportExportTest, importExportedFunctionParameterResultOnly) {
    if (target_device == ov::test::utils::DEVICE_MULTI || target_device == ov::test::utils::DEVICE_AUTO) {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    // Create a simple function
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(elementType, ov::Shape({1, 3, 24, 24}));
        param->set_friendly_name("param");
        param->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                      ov::ParameterVector{param});
        function->set_friendly_name("ParamResult");
    }

    auto execNet = core->compile_model(function, target_device, configuration);
    std::stringstream strm;
    execNet.export_model(strm);

    ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->inputs().size(), importedCompiledModel.inputs().size());
    EXPECT_NO_THROW(importedCompiledModel.input());
    EXPECT_NO_THROW(importedCompiledModel.input("data").get_node());
    EXPECT_THROW(importedCompiledModel.input("param"), ov::Exception);

    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), importedCompiledModel.outputs().size());
    EXPECT_NO_THROW(importedCompiledModel.output());
    EXPECT_EQ(function->output(0).get_tensor().get_names(),
              importedCompiledModel.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedCompiledModel.output("data").get_node());
    EXPECT_THROW(importedCompiledModel.output("param"), ov::Exception);

    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.input("data").get_element_type());
    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("data").get_element_type());
}

TEST_P(OVExecGraphImportExportTest, importExportedFunctionConstantResultOnly) {
    if (target_device == ov::test::utils::DEVICE_MULTI || target_device == ov::test::utils::DEVICE_AUTO) {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    // Create a simple function
    {
        auto constant = std::make_shared<ov::op::v0::Constant>(elementType, ov::Shape({1, 3, 24, 24}));
        constant->set_friendly_name("constant");
        constant->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::op::v0::Result>(constant);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                      ov::ParameterVector{});
        function->set_friendly_name("ConstResult");
    }

    auto execNet = core->compile_model(function, target_device, configuration);
    std::stringstream strm;
    execNet.export_model(strm);

    ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 0);
    EXPECT_EQ(function->inputs().size(), importedCompiledModel.inputs().size());
    EXPECT_THROW(importedCompiledModel.input(), ov::Exception);
    EXPECT_THROW(importedCompiledModel.input("data"), ov::Exception);
    EXPECT_THROW(importedCompiledModel.input("constant"), ov::Exception);

    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), importedCompiledModel.outputs().size());
    EXPECT_NO_THROW(importedCompiledModel.output());
    EXPECT_EQ(function->output(0).get_tensor().get_names(),
              importedCompiledModel.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedCompiledModel.output("data").get_node());
    EXPECT_THROW(importedCompiledModel.output("constant"), ov::Exception);

    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("data").get_element_type());
}

TEST_P(OVExecGraphImportExportTest, readFromV10IR) {
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
    function = core->read_model(model, ov::Tensor());
    EXPECT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_NO_THROW(function->input("in1"));     // remove if read_model does not change function names
    EXPECT_NO_THROW(function->input("in2"));     // remove if read_model does not change function names
    EXPECT_NO_THROW(function->output("concat"));  // remove if read_model does not change function names

    ov::CompiledModel execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(execNet.inputs().size(), 2);
    EXPECT_EQ(execNet.outputs().size(), 1);
    EXPECT_NO_THROW(execNet.input("in1"));
    EXPECT_NO_THROW(execNet.input("in2"));
    EXPECT_NO_THROW(execNet.output("concat"));

    if (target_device == ov::test::utils::DEVICE_MULTI || target_device == ov::test::utils::DEVICE_AUTO) {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    std::stringstream strm;
    execNet.export_model(strm);

    ov::CompiledModel importedExecNet = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(importedExecNet.inputs().size(), 2);
    EXPECT_EQ(importedExecNet.outputs().size(), 1);
    EXPECT_NO_THROW(importedExecNet.input("in1"));
    EXPECT_NO_THROW(importedExecNet.input("in2"));
    EXPECT_NO_THROW(importedExecNet.output("concat"));

    EXPECT_EQ(importedExecNet.input("in1").get_element_type(), ov::element::f32);
    EXPECT_EQ(importedExecNet.input("in2").get_element_type(), ov::element::f32);
    EXPECT_EQ(importedExecNet.output().get_element_type(), ov::element::f32);
}

static std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    auto to_config_string = [] (const Any& any) -> std::string {
        if (any.is<bool>()) {
            return any.as<bool>() ? "YES" : "NO";
        } else {
            std::stringstream strm;
            any.print(strm);
            return strm.str();
        }
    };
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, to_config_string(value.second));
    }
    return result;
}

std::string OVExecGraphUniqueNodeNames::getTestCaseName(testing::TestParamInfo<ov::test::BasicParams> obj) {
    ov::element::Type type;
    ov::Shape shape;
    std::string device;
    std::tie(type, shape, device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(shape) << "_";
    result << "inType=" << type.get_type_name() << "_";
    result << "targetDevice=" << device;
    return result.str();
}

void OVExecGraphUniqueNodeNames::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::element::Type type;
    ov::Shape shape;
    std::tie(type, shape, target_device) = this->GetParam();

    APIBaseTest::SetUp();

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, shape)};
    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto concat = std::make_shared<ov::op::v0::Concat>(split->outputs(), 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    fnPtr = std::make_shared<ov::Model>(results, params, "SplitConvConcat");
}

TEST_P(OVExecGraphUniqueNodeNames, CheckUniqueNodeNames) {
    auto core = utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(fnPtr, target_device);

    auto runtime_model = compiled_model.get_runtime_model();

    std::unordered_set<std::string> names;
    for (auto&& op : runtime_model->get_ops()) {
        ASSERT_TRUE(names.find(op->get_friendly_name()) == names.end()) << "Node with name " << op->get_friendly_name() << "already exists";
        names.insert(op->get_friendly_name());
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
