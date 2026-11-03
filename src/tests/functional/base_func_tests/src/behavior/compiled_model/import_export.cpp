// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/import_export.hpp"

#include <gmock/gmock-matchers.h>

#include <cstring>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov {
namespace test {
namespace behavior {
namespace hsm = ov::runtime::hsm;

std::string OVCompiledGraphImportExportTest::getTestCaseName(
    testing::TestParamInfo<OVCompiledGraphImportExportTestParams> obj) {
    const auto& [elementType, _targetDevice, configuration] = obj.param;
    auto targetDevice = _targetDevice;
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

void OVCompiledGraphImportExportTest::SetUp() {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(elementType, target_device, configuration) = this->GetParam();
    APIBaseTest::SetUp();
}

void OVCompiledGraphImportExportTest::TearDown() {
    if (!configuration.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

TEST_P(OVCompiledGraphImportExportTest, importExportedFunction) {
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
    EXPECT_EQ(function->input(0).get_element_type(), importedExecNet.input(0).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_tensor().get_names(), importedExecNet.input(1).get_tensor().get_names());
    EXPECT_EQ(function->input(1).get_tensor().get_partial_shape(),
              importedExecNet.input(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(1).get_tensor().get_element_type(),
              importedExecNet.input(1).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_element_type(), importedExecNet.input(1).get_tensor().get_element_type());
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
    EXPECT_EQ(function->output(0).get_element_type(), importedExecNet.output(0).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_tensor().get_names(), importedExecNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(),
              importedExecNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(),
              importedExecNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(function->output(1).get_element_type(), importedExecNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(importedExecNet.output(0).get_node(), importedExecNet.output("concat1").get_node());
    EXPECT_NE(importedExecNet.output(1).get_node(), importedExecNet.output("concat1").get_node());
    EXPECT_EQ(importedExecNet.output(1).get_node(), importedExecNet.output("concat2").get_node());
    EXPECT_NE(importedExecNet.output(0).get_node(), importedExecNet.output("concat2").get_node());
    EXPECT_THROW(importedExecNet.input("param1"), ov::Exception);
    EXPECT_THROW(importedExecNet.input("param2"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("result1"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("result2"), ov::Exception);
}

TEST_P(OVCompiledGraphImportExportTest, importExportedFunctionParameterResultOnly) {
    // Create a simple function
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(elementType, ov::Shape({1, 3, 24, 24}));
        param->set_friendly_name("param");
        param->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
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
    EXPECT_EQ(function->output(0).get_tensor().get_names(), importedCompiledModel.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedCompiledModel.output("data").get_node());
    EXPECT_THROW(importedCompiledModel.output("param"), ov::Exception);

    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.input("data").get_element_type());
    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("data").get_element_type());
}

TEST_P(OVCompiledGraphImportExportTest, importExportedFunctionConstantResultOnly) {
    // Create a simple function
    {
        auto constant = std::make_shared<ov::op::v0::Constant>(elementType, ov::Shape({1, 3, 24, 24}));
        constant->set_friendly_name("constant");
        constant->output(0).get_tensor().set_names({"data"});
        auto result = std::make_shared<ov::op::v0::Result>(constant);
        result->set_friendly_name("result");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
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
    EXPECT_EQ(function->output(0).get_tensor().get_names(), importedCompiledModel.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedCompiledModel.output("data").get_node());
    EXPECT_THROW(importedCompiledModel.output("constant"), ov::Exception);

    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("data").get_element_type());
}

TEST_P(OVCompiledGraphImportExportTest, readFromV10IR) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset8">
            <data element_type="f16" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP16" names="data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="round" id="1" type="Round" version="opset8">
            <data mode="half_to_even"/>
            <input>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="r">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset8">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    function = core->read_model(model, ov::Tensor());
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_NO_THROW(function->input("in1"));     // remove if read_model does not change function names
    EXPECT_NO_THROW(function->output("round"));  // remove if read_model does not change function names

    ov::CompiledModel execNet = core->compile_model(function, target_device, configuration);
    EXPECT_EQ(execNet.inputs().size(), 1);
    EXPECT_EQ(execNet.outputs().size(), 1);
    EXPECT_NO_THROW(execNet.input("in1"));
    EXPECT_NO_THROW(execNet.output("round"));

    std::stringstream strm;
    execNet.export_model(strm);

    ov::CompiledModel importedExecNet = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(importedExecNet.inputs().size(), 1);
    EXPECT_EQ(importedExecNet.outputs().size(), 1);
    EXPECT_NO_THROW(importedExecNet.input("in1"));
    EXPECT_NO_THROW(importedExecNet.output("round"));

    EXPECT_EQ(importedExecNet.input().get_element_type(), ov::element::f32);
    EXPECT_EQ(importedExecNet.output().get_element_type(), ov::element::f32);
}

TEST_P(OVCompiledGraphImportExportTest, importExportedFunctionDoubleInputOutput) {
    ov::CompiledModel compiledModel;

    // Create simple function
    function = ov::test::utils::make_multiple_input_output_double_concat({1, 2, 24, 24}, elementType);
    compiledModel = core->compile_model(function, target_device, configuration);

    std::stringstream strm;
    compiledModel.export_model(strm);

    ov::CompiledModel importedCompiledModel = core->import_model(strm, target_device, configuration);
    EXPECT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), importedCompiledModel.inputs().size());
    EXPECT_NO_THROW(importedCompiledModel.input("data1").get_node());
    EXPECT_NO_THROW(importedCompiledModel.input("data2").get_node());

    EXPECT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), importedCompiledModel.outputs().size());
    EXPECT_NO_THROW(importedCompiledModel.output("concat1"));
    EXPECT_NO_THROW(importedCompiledModel.output("concat2"));

    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.input("data1").get_element_type());
    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.input("data2").get_element_type());
    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("concat2").get_element_type());
    EXPECT_EQ(ov::element::Type(elementType), importedCompiledModel.output("concat1").get_element_type());
}

//
// ImportExportNetwork
//

TEST_P(OVClassCompiledModelImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    ov::Core ie = ov::test::utils::create_core();
    std::stringstream strm;
    ov::CompiledModel executableNetwork;
    OV_ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, target_device));
    OV_ASSERT_NO_THROW(executableNetwork.export_model(strm));
    OV_ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, target_device));
    OV_ASSERT_NO_THROW(executableNetwork.create_infer_request());
}

TEST_P(OVClassCompiledModelImportExportTestP, smoke_ImportNetworkThrowWithDeviceName) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core ie = ov::test::utils::create_core();
    std::stringstream wrongStm;
    // Plugins with their own blob header (CPU/Hetero) reject it there; others fall back to the generic
    // Core::read_model rejection.
    OV_EXPECT_THROW((ie.import_model(wrongStm, target_device)),
                    ov::Exception,
                    testing::AnyOf(testing::HasSubstr("device xml header"),
                                   testing::HasSubstr("requirements header"),
                                   testing::HasSubstr("Unable to read the model")));
}

//
// GetRuntimeModel
//

std::string OVCompiledModelGraphUniqueNodeNamesTest::getTestCaseName(
    testing::TestParamInfo<OVCompiledModelGraphUniqueNodeNamesTestParams> obj) {
    const auto& [netPrecision, inputShapes, _targetDevice] = obj.param;
    auto targetDevice = _targetDevice;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.to_string() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void OVCompiledModelGraphUniqueNodeNamesTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [netPrecision, inputShape, _target_device] = this->GetParam();
    target_device = _target_device;

    APIBaseTest::SetUp();

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape(inputShape))};
    auto split_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto concat = std::make_shared<ov::op::v0::Concat>(split->outputs(), 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    model = std::make_shared<ov::Model>(results, params, "SplitConvConcat");
}

TEST_P(OVCompiledModelGraphUniqueNodeNamesTest, CheckUniqueNodeNames) {
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model, target_device);
    auto exec_graph = compiled_model.get_runtime_model();

    std::unordered_set<std::string> names;
    ASSERT_NE(exec_graph, nullptr);

    for (const auto& op : exec_graph->get_ops()) {
        ASSERT_TRUE(names.find(op->get_friendly_name()) == names.end())
            << "Node with name " << op->get_friendly_name() << "already exists";
        names.insert(op->get_friendly_name());

        const auto& rtInfo = op->get_rt_info();
        auto it = rtInfo.find(ov::exec_model_info::LAYER_TYPE);
        ASSERT_NE(rtInfo.end(), it);
    }
};

const char serialize_test_model[] = R"V0G0N(<?xml version="1.0" ?>
<?xml version="1.0" ?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="A" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add_node1" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_node2" type="Multiply" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="add_node3" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="C" type="Parameter" version="opset1">
			<data element_type="f32" shape="1"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="add_node4" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="Y" type="Add" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="Y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="6" to-port="1"/>
		<edge from-layer="6" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
	</edges>
</net>
)V0G0N";

const char expected_serialized_model[] = R"V0G0N(
<?xml version="1.0"?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="C" type="Input">
			<data shape="1" element_type="f32" execOrder="3" execTimeMcs="not_executed" originalLayersNames="C" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Input">
			<data shape="1" element_type="f32" execOrder="1" execTimeMcs="not_executed" originalLayersNames="B" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="A" type="Input">
			<data shape="1" element_type="f32" execOrder="0" execTimeMcs="not_executed" originalLayersNames="A" outputLayouts="x" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="add_node2" type="Eltwise">
			<data execOrder="2" execTimeMcs="not_executed" originalLayersNames="add_node2" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="add_node1" type="Eltwise">
			<data execOrder="4" execTimeMcs="not_executed" originalLayersNames="add_node1,add_node3,add_node4" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Y" type="Eltwise">
			<data execOrder="5" execTimeMcs="not_executed" originalLayersNames="Y" outputLayouts="x" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="out_Y" type="Output">
			<data execOrder="6" execTimeMcs="not_executed" originalLayersNames="" outputLayouts="undef" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32"/>
			<input>
				<port id="0">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="3" />
		<edge from-layer="0" from-port="0" to-layer="5" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="1" from-port="0" to-layer="4" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="2" from-port="0" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="2" to-layer="4" to-port="2" />
		<edge from-layer="4" from-port="4" to-layer="5" to-port="0" />
		<edge from-layer="5" from-port="2" to-layer="6" to-port="0" />
	</edges>
</net>
)V0G0N";

const char expected_serialized_model_cpu[] = R"V0G0N(
<?xml version="1.0"?>
<net name="addmul_abc" version="10">
	<layers>
		<layer id="0" name="C" type="Input">
			<data shape="1" element_type="f32" execOrder="2" execTimeMcs="not_executed" originalLayersNames="C" outputLayouts="a" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="B" type="Input">
			<data shape="1" element_type="f32" execOrder="1" execTimeMcs="not_executed" originalLayersNames="B" outputLayouts="a" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="A" type="Input">
			<data shape="1" element_type="f32" execOrder="0" execTimeMcs="not_executed" originalLayersNames="A" outputLayouts="a" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Y" type="Subgraph">
			<data execOrder="3" execTimeMcs="not_executed" originalLayersNames="add_node1,add_node2,add_node3,add_node4,Y" outputLayouts="a" outputPrecisions="FP32" primitiveType="jit_avx512_FP32" runtimePrecision="FP32" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>1</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Y/sink_port_0" type="Output">
			<data execOrder="4" execTimeMcs="not_executed" originalLayersNames="Y/sink_port_0" outputLayouts="undef" outputPrecisions="FP32" primitiveType="unknown_FP32" runtimePrecision="FP32" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="2" />
		<edge from-layer="0" from-port="0" to-layer="3" to-port="3" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="3" from-port="4" to-layer="4" to-port="0" />
	</edges>
	<rt_info />
</net>
)V0G0N";

std::string OVExecGraphSerializationTest::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    result << "TargetDevice=" << target_device;
    return result.str();
}

void OVExecGraphSerializationTest::SetUp() {
    target_device = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();

    const std::string XML_EXT = ".xml";
    const std::string BIN_EXT = ".bin";

    std::string filePrefix = ov::test::utils::generateTestFilePrefix();

    m_out_xml_path = filePrefix + XML_EXT;
    m_out_bin_path = filePrefix + BIN_EXT;
}

void OVExecGraphSerializationTest::TearDown() {
    APIBaseTest::TearDown();
    ov::test::utils::removeIRFiles(m_out_xml_path, m_out_bin_path);
}

bool OVExecGraphSerializationTest::exec_graph_walker::for_each(pugi::xml_node& node) {
    std::string node_name{node.name()};
    if (node_name == "layer" || node_name == "data") {
        nodes.push_back(node);
    }
    return true;  // continue traversal
}

std::pair<bool, std::string> OVExecGraphSerializationTest::compare_nodes(const pugi::xml_node& node1,
                                                                         const pugi::xml_node& node2) {
    // node names must be the same
    const std::string node1_name{node1.name()};
    const std::string node2_name{node2.name()};
    if (node1_name != node2_name) {
        return {false, "Node name is different: " + node1_name + " != " + node2_name};
    }

    // node attribute count must be the same
    const auto attr1 = node1.attributes();
    const auto attr2 = node2.attributes();
    const auto attr1_size = std::distance(attr1.begin(), attr1.end());
    const auto attr2_size = std::distance(attr2.begin(), attr2.end());
    if (attr1_size != attr2_size) {
        return {false,
                "Attribute count is different in <" + node1_name + "> :" + std::to_string(attr1_size) +
                    " != " + std::to_string(attr2_size)};
    }

    // every node attribute name must be the same
    auto a1 = attr1.begin();
    auto a2 = attr2.begin();
    for (int j = 0; j < attr1_size; ++j, ++a1, ++a2) {
        const std::string a1_name{a1->name()};
        const std::string a2_name{a2->name()};
        const std::string a1_value{a1->value()};
        const std::string a2_value{a2->value()};
        if (a1_name != a2_name || (a1_name == "type" && a1_value != a2_value)) {
            // TODO: Remove temporary w/a later
            if (a1_value == "Output" && a2_value == "Result") {
                continue;
            }
            return {false,
                    "Attributes are different in <" + node1_name + "> : " + a1_name + "=" + a1_value +
                        " != " + a2_name + "=" + a2_value};
        }
    }
    return {true, ""};
}

std::pair<bool, std::string> OVExecGraphSerializationTest::compare_docs(const pugi::xml_document& doc1,
                                                                        const pugi::xml_document& doc2) {
    // traverse document and prepare vector of <layer> & <data> nodes to compare
    exec_graph_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    // nodes count must be the same
    const auto& nodes1 = walker1.nodes;
    const auto& nodes2 = walker2.nodes;
    if (nodes1.size() != nodes2.size()) {
        return {false, "Node count differ: " + std::to_string(nodes1.size()) + " != " + std::to_string(nodes2.size())};
    }

    // every node must be equivalent
    for (int i = 0; i < nodes1.size(); i++) {
        const auto res = compare_nodes(nodes1[i], nodes2[i]);
        if (res.first == false) {
            return res;
        }
    }
    return {true, ""};
}

TEST_P(OVExecGraphSerializationTest, ExecutionGraph) {
    auto core = utils::PluginCache::get().core();
    auto model = core->read_model(serialize_test_model);
    auto compiled_model = core->compile_model(model, target_device);
    auto runtime_model = compiled_model.get_runtime_model();

    ov::serialize(runtime_model, m_out_xml_path, m_out_bin_path);

    pugi::xml_document expected;
    pugi::xml_document result;
    if (target_device == "CPU" || target_device == "AUTO:CPU" || target_device == "MULTI:CPU") {
        ASSERT_TRUE(expected.load_string(expected_serialized_model_cpu));
    } else {
        ASSERT_TRUE(expected.load_string(expected_serialized_model));
    }
    ASSERT_TRUE(result.load_file(m_out_xml_path.c_str()));

    const auto& [status, message] = this->compare_docs(expected, result);

    ASSERT_TRUE(status) << message;
}

std::string hsm_repack(const std::string& blob, const std::function<void(std::vector<HsmEntry>&)>& modify) {
    const auto data = reinterpret_cast<const std::byte*>(blob.data());
    const auto reader = hsm::Reader::open(data, blob.size());
    OPENVINO_ASSERT(reader, "Not an HSM blob");

    std::vector<HsmEntry> entries;
    for (const auto& entry : reader->entries()) {
        // Built directly from the entry, so duplicated (device, tag) pairs are kept too.
        const auto section = entry.tag.is_inline()
                                 ? hsm::Section{entry}
                                 : hsm::Section{entry, {data + entry.offset, static_cast<size_t>(entry.size)}};
        entries.push_back({entry.device, {entry.tag, entry.tag_reserved}, section.to_bytes().value()});
    }
    modify(entries);

    std::stringstream out;
    auto writer = hsm::DeferredWriter::open(out);
    OPENVINO_ASSERT(writer, "Cannot open HSM writer");
    for (const auto& entry : entries) {
        OPENVINO_ASSERT(
            writer->add_section(entry.device, entry.tag, {entry.payload.data(), entry.payload.size()}, {4096}),
            "Cannot add HSM section");
    }
    OPENVINO_ASSERT(!writer->finalize(), "HSM repack failed");
    return out.str();
}

std::string hsm_with_minor_version(std::string blob, uint8_t minor) {
    OPENVINO_ASSERT(blob.size() >= sizeof(hsm::Header), "Blob too small for an HSM header");
    blob[offsetof(hsm::Header, version_minor)] = static_cast<char>(minor);
    return blob;
}

bool is_hsm(const std::string& blob) {
    return blob.size() >= sizeof(hsm::Header) && hsm::Header::view(blob.data()).magic == hsm::BlobMagic::single;
}

namespace {
ov::Tensor hsm_infer(ov::CompiledModel& compiled, const ov::Tensor& input) {
    auto request = compiled.create_infer_request();
    request.set_input_tensor(input);
    request.infer();
    const auto output = request.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    output.copy_to(result);
    return result;
}

void add_unknown_sections(std::vector<HsmEntry>& entries) {
    const auto bytes = [](std::string_view text, size_t size) {
        std::vector<std::byte> payload(size);
        std::memcpy(payload.data(), text.data(), std::min(size, text.size()));
        return payload;
    };
    // A Core tag id not assigned yet, another device's pointer section and another device's inline section.
    entries.push_back({hsm::any_device_id,
                       hsm::SectionTag::make(hsm::core_tag_id_range_end - 1, false),
                       bytes("future core section", 19)});
    entries.push_back({250, hsm::SectionTag::make_device_tag(0x123, false), bytes("other device", 12)});
    entries.push_back({251, hsm::SectionTag::make_device_tag(0, true), bytes("inline", 24)});
}
}  // namespace

std::string OVCompiledModelHsmBlobTest::getTestCaseName(const testing::TestParamInfo<HsmBlobTestParams>& obj) {
    auto [device, configuration] = obj.param;
    std::replace(device.begin(), device.end(), ':', '.');
    std::ostringstream result;
    result << "targetDevice=" << device;
    for (const auto& [key, value] : configuration) {
        result << "_" << key << "=" << value.as<std::string>();
    }
    return result.str();
}

void OVCompiledModelHsmBlobTest::SetUp() {
    std::tie(target_device, configuration) = GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    APIBaseTest::SetUp();

    core = ov::test::utils::create_core();
    // Plugin-level too: Core picks the cache blob format from the plugin, not from the compile config.
    core.set_property(target_device, configuration);
    ASSERT_TRUE(core.get_property(target_device, ov::internal::emit_hsm_format));

    model = ov::test::utils::make_conv_pool_relu({1, 3, 16, 16}, ov::element::f32);
    cache_path = "hsm_blob_cache_" + ov::test::utils::generateTestFilePrefix();
    input = ov::test::utils::create_and_fill_tensor(ov::element::f32, model->input().get_shape());
    auto compiled = compile(false);
    reference = hsm_infer(compiled, input);
}

void OVCompiledModelHsmBlobTest::TearDown() {
    if (!cache_path.empty()) {
        ov::test::utils::removeFilesWithExt(ov::util::path_to_string(cache_path), "blob");
        ov::test::utils::removeDir(ov::util::path_to_string(cache_path));
    }
    APIBaseTest::TearDown();
}

ov::CompiledModel OVCompiledModelHsmBlobTest::compile(bool with_cache) {
    auto config = configuration;
    if (with_cache) {
        config.emplace(ov::cache_path(cache_path));
    }
    return core.compile_model(model, target_device, config);
}

std::string OVCompiledModelHsmBlobTest::export_blob(ov::CompiledModel compiled) const {
    std::stringstream stream;
    compiled.export_model(stream);
    return stream.str();
}

std::string OVCompiledModelHsmBlobTest::read_cache_blob() const {
    const auto blobs = ov::test::utils::listFilesWithExt(ov::util::path_to_string(cache_path), "blob");
    OPENVINO_ASSERT(blobs.size() == 1, "Expected exactly one cache blob, got ", blobs.size());
    std::ifstream file(blobs.front(), std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void OVCompiledModelHsmBlobTest::write_cache_blob(const std::string& blob) const {
    const auto blobs = ov::test::utils::listFilesWithExt(ov::util::path_to_string(cache_path), "blob");
    ASSERT_EQ(blobs.size(), 1u) << "Expected exactly one cache blob";
    // Cache files may be created read-only, so replace the file instead of overwriting it.
    std::filesystem::remove(blobs.front());
    std::ofstream file(blobs.front(), std::ios::binary);
    file.write(blob.data(), blob.size());
    ASSERT_TRUE(file.good()) << "Cannot write cache blob " << blobs.front();
}

void OVCompiledModelHsmBlobTest::expect_same_inference(ov::CompiledModel& actual) {
    ov::test::utils::compare(reference, hsm_infer(actual, input));
}

TEST_P(OVCompiledModelHsmBlobTest, cache_blob_is_identical_to_exported_blob) {
    auto compiled = compile(true);
    const auto exported = export_blob(compiled);

    EXPECT_TRUE(is_hsm(exported));
    EXPECT_EQ(read_cache_blob(), exported);
}

TEST_P(OVCompiledModelHsmBlobTest, cache_blob_is_imported_via_import_model) {
    compile(true);
    const auto blob = read_cache_blob();

    std::stringstream stream(blob);
    auto from_stream = core.import_model(stream, target_device, configuration);
    expect_same_inference(from_stream);

    ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
    std::memcpy(tensor.data(), blob.data(), blob.size());
    auto from_tensor = core.import_model(tensor, target_device, configuration);
    expect_same_inference(from_tensor);
}

TEST_P(OVCompiledModelHsmBlobTest, exported_blob_is_loaded_from_cache) {
    // The extra section makes the exported blob distinguishable from what the cache itself would write.
    const auto exported = hsm_repack(export_blob(compile(false)), add_unknown_sections);
    compile(true);
    write_cache_blob(exported);

    auto cached = compile(true);

    EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
    EXPECT_EQ(read_cache_blob(), exported);
    expect_same_inference(cached);
}

TEST_P(OVCompiledModelHsmBlobTest, unknown_sections_are_skipped_on_import) {
    std::stringstream stream(hsm_repack(export_blob(compile(false)), add_unknown_sections));

    auto imported = core.import_model(stream, target_device, configuration);

    expect_same_inference(imported);
}

TEST_P(OVCompiledModelHsmBlobTest, newer_minor_version_is_imported) {
    const auto newer = hsm_with_minor_version(hsm_repack(export_blob(compile(false)), add_unknown_sections),
                                              hsm::FormatVersion::minor + 1);
    std::stringstream stream(newer);

    auto imported = core.import_model(stream, target_device, configuration);

    expect_same_inference(imported);
}

TEST_P(OVCompiledModelHsmBlobTest, exported_blob_is_a_valid_container_of_the_current_version) {
    const auto exported = export_blob(compile(false));
    const auto data = reinterpret_cast<const std::byte*>(exported.data());

    EXPECT_TRUE(hsm::ContainerView(data, exported.size()).validate());
    EXPECT_TRUE(hsm::Reader::open(data, exported.size()));
    const auto& header = hsm::Header::view(exported.data());
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);
    EXPECT_EQ(header.container_size, exported.size());
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OVClassCompiledModelImportExportTestP);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OVCompiledModelHsmBlobTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OVCompiledModelGraphUniqueNodeNamesTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OVExecGraphSerializationTest);

}  // namespace behavior
}  // namespace test
}  // namespace ov
