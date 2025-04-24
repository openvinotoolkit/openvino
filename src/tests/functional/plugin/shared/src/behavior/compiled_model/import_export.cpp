// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include "behavior/compiled_model/import_export.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"

#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVCompiledGraphImportExportTest::getTestCaseName(testing::TestParamInfo<OVCompiledGraphImportExportTestParams> obj) {
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

void  OVCompiledGraphImportExportTest::SetUp() {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(elementType, target_device, configuration) = this->GetParam();
    APIBaseTest::SetUp();
}

void  OVCompiledGraphImportExportTest::TearDown() {
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

TEST_P(OVCompiledGraphImportExportTest, importExportedFunctionParameterResultOnly) {
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

TEST_P(OVCompiledGraphImportExportTest, importExportedFunctionConstantResultOnly) {
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
    ov::Core ie = createCoreWithTemplate();
    std::stringstream wrongStm;
    // Import model with wrong format throws exception
    OV_EXPECT_THROW((ie.import_model(wrongStm, target_device)),
                    ov::Exception,
                    testing::HasSubstr("device xml header"));
}

//
// GetRuntimeModel
//

std::string OVCompiledModelGraphUniqueNodeNamesTest::getTestCaseName(testing::TestParamInfo<OVCompiledModelGraphUniqueNodeNamesTestParams> obj) {
    ov::element::Type netPrecision;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.to_string() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void OVCompiledModelGraphUniqueNodeNamesTest::SetUp() {
    ov::Shape inputShape;
    ov::element::Type netPrecision;
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(netPrecision, inputShape, target_device) = this->GetParam();

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

bool OVExecGraphSerializationTest::exec_graph_walker::for_each(pugi::xml_node &node) {
    std::string node_name{node.name()};
    if (node_name == "layer" || node_name == "data") {
        nodes.push_back(node);
    }
    return true;  // continue traversal
}

std::pair<bool, std::string> OVExecGraphSerializationTest::compare_nodes(const pugi::xml_node &node1,
                                                                         const pugi::xml_node &node2) {
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
        return {false, "Attribute count is different in <" + node1_name + "> :" +
                       std::to_string(attr1_size) + " != " +
                       std::to_string(attr2_size)};
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
            return {false, "Attributes are different in <" + node1_name + "> : " +
                           a1_name + "=" + a1_value + " != " + a2_name +
                           "=" + a2_value};
        }
    }
    return {true, ""};
}

std::pair<bool, std::string> OVExecGraphSerializationTest::compare_docs(const pugi::xml_document &doc1,
                                                                      const pugi::xml_document &doc2) {
    // traverse document and prepare vector of <layer> & <data> nodes to compare
    exec_graph_walker walker1, walker2;
    doc1.child("net").child("layers").traverse(walker1);
    doc2.child("net").child("layers").traverse(walker2);

    // nodes count must be the same
    const auto &nodes1 = walker1.nodes;
    const auto &nodes2 = walker2.nodes;
    if (nodes1.size() != nodes2.size()) {
        return {false, "Node count differ: " + std::to_string(nodes1.size()) +
                       " != " + std::to_string(nodes2.size())};
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

    bool status;
    std::string message;
    std::tie(status, message) = this->compare_docs(expected, result);

    ASSERT_TRUE(status) << message;
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
