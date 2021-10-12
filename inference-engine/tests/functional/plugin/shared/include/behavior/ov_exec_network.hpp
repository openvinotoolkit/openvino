// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "cpp/ie_cnn_network.h"
#include "ie_core.hpp"
#include "ie_ngraph_utils.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/runtime.hpp"
#include "transformations/serialize.hpp"
#include "functional_test_utils/plugin_cache.hpp"

namespace ov {
namespace test {
using OVExecNetwork = ov::test::BehaviorTestsBasic;

TEST_P(OVExecNetwork, getInputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->inputs().size(), execNet.inputs().size());
    EXPECT_NO_THROW(execNet.input());
    EXPECT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
    EXPECT_EQ(function->input().get_tensor().get_partial_shape(), execNet.input().get_tensor().get_partial_shape());
    EXPECT_EQ(function->input().get_tensor().get_element_type(), execNet.input().get_tensor().get_element_type());
}

TEST_P(OVExecNetwork, getOutputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    execNet = ie->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), execNet.outputs().size());
    EXPECT_NO_THROW(execNet.output());
    EXPECT_EQ(function->output().get_tensor().get_names(), execNet.output().get_tensor().get_names());
    EXPECT_EQ(function->output().get_tensor().get_partial_shape(), execNet.output().get_tensor().get_partial_shape());
    EXPECT_EQ(function->output().get_tensor().get_element_type(), execNet.output().get_tensor().get_element_type());
}

TEST_P(OVExecNetwork, getInputsFromFunctionWithSeveralInputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
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
        function->set_friendly_name("SingleRuLU");
    }
    execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_EQ(function->inputs().size(), 2);
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

TEST_P(OVExecNetwork, getOutputsFromFunctionWithSeveralOutputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
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
        function->set_friendly_name("SingleRuLU");
    }
    execNet = ie->compile_model(function, targetDevice, configuration);
    ASSERT_EQ(function->outputs().size(), 2);
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
}

TEST_P(OVExecNetwork, precisionsAsInOriginalFunction) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;
    execNet = ie->compile_model(function, targetDevice, configuration);

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
TEST_P(OVExecNetwork, precisionsAsInOriginalIR) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const std::string m_out_xml_path_1 = "precisionsAsInOriginalIR.xml";
    const std::string m_out_bin_path_1 = "precisionsAsInOriginalIR.bin";
    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(function);

    ov::runtime::ExecutableNetwork execNet;
    execNet = ie->compile_model(m_out_xml_path_1, targetDevice, configuration);

    EXPECT_EQ(function->get_parameters().size(), execNet.inputs().size());
    auto ref_parameter = function->get_parameters().back();
    auto actual_parameter = execNet.inputs().back().get_node_shared_ptr();
    EXPECT_EQ(elementType, ref_parameter->get_element_type());
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

TEST_P(OVExecNetwork, importExportedFunction) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (targetDevice == "MULTI" || targetDevice == "AUTO") {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
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
        function->set_friendly_name("SingleRuLU");
    }
    execNet = ie->compile_model(function, targetDevice, configuration);

    std::stringstream strm;
    execNet.export_model(strm);

    ov::runtime::ExecutableNetwork importedExecNet = ie->import_model(strm, targetDevice, configuration);
    ASSERT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), importedExecNet.inputs().size());
    EXPECT_THROW(importedExecNet.input(), ov::Exception);
    EXPECT_EQ(function->input(0).get_tensor().get_names(), importedExecNet.input(0).get_tensor().get_names());
    EXPECT_EQ(function->input(0).get_tensor().get_partial_shape(),
              importedExecNet.input(0).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(0).get_tensor().get_element_type(),
              importedExecNet.input(0).get_tensor().get_element_type());
    EXPECT_EQ(function->input(1).get_tensor().get_names(), importedExecNet.input(1).get_tensor().get_names());
    EXPECT_EQ(function->input(1).get_tensor().get_partial_shape(),
              importedExecNet.input(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->input(1).get_tensor().get_element_type(),
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
    EXPECT_EQ(function->output(1).get_tensor().get_names(), importedExecNet.output(1).get_tensor().get_names());
    EXPECT_EQ(function->output(1).get_tensor().get_partial_shape(),
              importedExecNet.output(1).get_tensor().get_partial_shape());
    EXPECT_EQ(function->output(1).get_tensor().get_element_type(),
              importedExecNet.output(1).get_tensor().get_element_type());
    EXPECT_EQ(importedExecNet.output(0).get_node(), importedExecNet.output("relu").get_node());
    EXPECT_NE(importedExecNet.output(1).get_node(), importedExecNet.output("relu").get_node());
    EXPECT_EQ(importedExecNet.output(1).get_node(), importedExecNet.output("concat").get_node());
    EXPECT_NE(importedExecNet.output(0).get_node(), importedExecNet.output("concat").get_node());
    EXPECT_THROW(importedExecNet.input("param1"), ov::Exception);
    EXPECT_THROW(importedExecNet.input("param2"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("concat_op"), ov::Exception);
    EXPECT_THROW(importedExecNet.output("relu_op"), ov::Exception);
}

TEST_P(OVExecNetwork, readFromV10IR) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

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
    function = ie->read_model(model, InferenceEngine::Blob::Ptr());
    EXPECT_EQ(function->inputs().size(), 1);
    EXPECT_EQ(function->outputs().size(), 1);
    EXPECT_NO_THROW(function->input("in1")); // remove if read_model does not change function names
    EXPECT_NO_THROW(function->output("round")); // remove if read_model does not change function names

    ov::runtime::ExecutableNetwork execNet = ie->compile_model(function, targetDevice, configuration);
    EXPECT_EQ(execNet.inputs().size(), 1);
    EXPECT_EQ(execNet.outputs().size(), 1);
    EXPECT_NO_THROW(execNet.input("in1"));
    EXPECT_NO_THROW(execNet.output("round"));

    if (targetDevice == "MULTI" || targetDevice == "AUTO") {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    std::stringstream strm;
    execNet.export_model(strm);

    ov::runtime::ExecutableNetwork importedExecNet = ie->import_model(strm, targetDevice, configuration);
    EXPECT_EQ(importedExecNet.inputs().size(), 1);
    EXPECT_EQ(importedExecNet.outputs().size(), 1);
    EXPECT_NO_THROW(importedExecNet.input("in1"));
    EXPECT_NO_THROW(importedExecNet.output("round"));

    EXPECT_EQ(importedExecNet.input().get_element_type(), ov::element::f32);
    EXPECT_EQ(importedExecNet.output().get_element_type(), ov::element::f32);
}

TEST_P(OVExecNetwork, importExportedIENetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    if (targetDevice == "MULTI" || targetDevice == "AUTO") {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    std::shared_ptr<InferenceEngine::Core> core = ::PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
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
        function->set_friendly_name("SingleRuLU");
    }
    execNet = core->LoadNetwork(InferenceEngine::CNNNetwork(function), targetDevice, configuration);

    std::stringstream strm;
    execNet.Export(strm);

    ov::runtime::ExecutableNetwork importedExecNet = ie->import_model(strm, targetDevice, configuration);
    ASSERT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), importedExecNet.inputs().size());
    EXPECT_THROW(importedExecNet.input(), ov::Exception);
    EXPECT_NO_THROW(importedExecNet.input("data1").get_node());
    EXPECT_NO_THROW(importedExecNet.input("data2").get_node());
    EXPECT_NO_THROW(importedExecNet.input("param1").get_node());
    EXPECT_NO_THROW(importedExecNet.input("param2").get_node());
    ASSERT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), importedExecNet.outputs().size());
    EXPECT_THROW(importedExecNet.output(), ov::Exception);
    EXPECT_NE(function->output(0).get_tensor().get_names(),
        importedExecNet.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedExecNet.output("relu").get_node());
    EXPECT_NO_THROW(importedExecNet.output("concat").get_node());
    EXPECT_NO_THROW(importedExecNet.output("relu_op").get_node());
    EXPECT_NO_THROW(importedExecNet.output("concat_op").get_node());

    const auto outputType = elementType == ngraph::element::i32 ||
        elementType == ngraph::element::i64 ? ngraph::element::i32 : ngraph::element::f32;
    const auto inputType = elementType == ngraph::element::f16 ? ngraph::element::f32 : elementType;

    EXPECT_EQ(inputType, importedExecNet.input("param1").get_element_type());
    EXPECT_EQ(inputType, importedExecNet.input("param2").get_element_type());
    EXPECT_EQ(outputType, importedExecNet.output("concat_op").get_element_type());
    EXPECT_EQ(outputType, importedExecNet.output("relu_op").get_element_type());
}


TEST_P(OVExecNetwork, ieImportExportedFunction) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    if (targetDevice == "MULTI" || targetDevice == "AUTO") {
        GTEST_SKIP() << "MULTI / AUTO does not support import / export" << std::endl;
    }

    std::shared_ptr<InferenceEngine::Core> core = ::PluginCache::get().ie();
    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
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
        function->set_friendly_name("SingleRuLU");
    }
    execNet = ie->compile_model(function, targetDevice, configuration);

    std::stringstream strm;
    execNet.export_model(strm);

    InferenceEngine::ExecutableNetwork importedExecNet = core->ImportNetwork(strm, targetDevice, configuration);
    ASSERT_EQ(function->inputs().size(), 2);
    EXPECT_EQ(function->inputs().size(), importedExecNet.GetInputsInfo().size());
    EXPECT_NO_THROW(importedExecNet.GetInputsInfo()["param1"]);
    EXPECT_NO_THROW(importedExecNet.GetInputsInfo()["param2"]);
    ASSERT_EQ(function->outputs().size(), 2);
    EXPECT_EQ(function->outputs().size(), importedExecNet.GetOutputsInfo().size());
    EXPECT_NO_THROW(importedExecNet.GetOutputsInfo()["relu_op"]);
    EXPECT_NO_THROW(importedExecNet.GetOutputsInfo()["concat_op"]);

    const auto prc = InferenceEngine::details::convertPrecision(elementType);

    EXPECT_EQ(prc, importedExecNet.GetInputsInfo()["param1"]->getPrecision());
    EXPECT_EQ(prc, importedExecNet.GetInputsInfo()["param2"]->getPrecision());
    EXPECT_EQ(prc, importedExecNet.GetOutputsInfo()["concat_op"]->getPrecision());
    EXPECT_EQ(prc, importedExecNet.GetOutputsInfo()["relu_op"]->getPrecision());
}

}  // namespace test
}  // namespace ov
