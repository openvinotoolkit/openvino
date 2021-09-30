// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/runtime.hpp"
#include "transformations/serialize.hpp"

namespace ov {
namespace test {
using OVExecNetwork = ov::test::BehaviorTestsBasic;

// Load correct network to Plugin to get executable network
TEST_P(OVExecNetwork, getInputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    ASSERT_EQ(function->inputs().size(), 1);
    ASSERT_EQ(function->inputs().size(), execNet.inputs().size());
    ASSERT_NO_THROW(execNet.input());
    ASSERT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
    ASSERT_EQ(function->input().get_tensor().get_partial_shape(), execNet.input().get_tensor().get_partial_shape());
    ASSERT_EQ(function->input().get_tensor().get_element_type(), execNet.input().get_tensor().get_element_type());
}

TEST_P(OVExecNetwork, getOutputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    ASSERT_EQ(function->outputs().size(), 1);
    ASSERT_EQ(function->outputs().size(), execNet.outputs().size());
    ASSERT_NO_THROW(execNet.output());
    ASSERT_EQ(function->output().get_tensor().get_names(), execNet.output().get_tensor().get_names());
    ASSERT_EQ(function->output().get_tensor().get_partial_shape(), execNet.output().get_tensor().get_partial_shape());
    ASSERT_EQ(function->output().get_tensor().get_element_type(), execNet.output().get_tensor().get_element_type());
}

TEST_P(OVExecNetwork, getInputsFromFunctionWithSeveralInputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param2->output(0).get_tensor().set_names({"data2"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->output(0).get_tensor().set_names({"relu"});
        auto result1 = std::make_shared<ov::opset8::Result>(relu);
        auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
        concat->output(0).get_tensor().set_names({"concat"});
        auto result2 = std::make_shared<ov::opset8::Result>(concat);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{param1, param2});
        function->set_friendly_name("SingleRuLU");
    }
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    ASSERT_EQ(function->inputs().size(), 2);
    ASSERT_EQ(function->inputs().size(), execNet.inputs().size());
    ASSERT_THROW(execNet.input(), ov::Exception);
    ASSERT_EQ(function->input(0).get_tensor().get_names(), execNet.input(0).get_tensor().get_names());
    ASSERT_EQ(function->input(0).get_tensor().get_partial_shape(), execNet.input(0).get_tensor().get_partial_shape());
    ASSERT_EQ(function->input(0).get_tensor().get_element_type(), execNet.input(0).get_tensor().get_element_type());
    ASSERT_EQ(function->input(1).get_tensor().get_names(), execNet.input(1).get_tensor().get_names());
    ASSERT_EQ(function->input(1).get_tensor().get_partial_shape(), execNet.input(1).get_tensor().get_partial_shape());
    ASSERT_EQ(function->input(1).get_tensor().get_element_type(), execNet.input(1).get_tensor().get_element_type());
    ASSERT_EQ(function->input(0).get_node(), function->input("data1").get_node());
    ASSERT_NE(function->input(1).get_node(), function->input("data1").get_node());
    ASSERT_EQ(function->input(1).get_node(), function->input("data2").get_node());
    ASSERT_NE(function->input(0).get_node(), function->input("data2").get_node());
}

TEST_P(OVExecNetwork, getOutputsFromFunctionWithSeveralOutputs) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;

    // Create simple function
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(elementType, ngraph::Shape({1, 3, 24, 24}));
        param2->output(0).get_tensor().set_names({"data2"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->output(0).get_tensor().set_names({"relu"});
        auto result1 = std::make_shared<ov::opset8::Result>(relu);
        auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
        concat->output(0).get_tensor().set_names({"concat"});
        auto result2 = std::make_shared<ov::opset8::Result>(concat);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{param1, param2});
        function->set_friendly_name("SingleRuLU");
    }
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    ASSERT_EQ(function->outputs().size(), 2);
    ASSERT_EQ(function->outputs().size(), execNet.outputs().size());
    ASSERT_THROW(execNet.output(), ov::Exception);
    ASSERT_EQ(function->output(0).get_tensor().get_names(), execNet.output(0).get_tensor().get_names());
    ASSERT_EQ(function->output(0).get_tensor().get_partial_shape(), execNet.output(0).get_tensor().get_partial_shape());
    ASSERT_EQ(function->output(0).get_tensor().get_element_type(), execNet.output(0).get_tensor().get_element_type());
    ASSERT_EQ(function->output(1).get_tensor().get_names(), execNet.output(1).get_tensor().get_names());
    ASSERT_EQ(function->output(1).get_tensor().get_partial_shape(), execNet.output(1).get_tensor().get_partial_shape());
    ASSERT_EQ(function->output(1).get_tensor().get_element_type(), execNet.output(1).get_tensor().get_element_type());
    ASSERT_EQ(function->output(0).get_node(), function->output("relu").get_node());
    ASSERT_NE(function->output(1).get_node(), function->output("relu").get_node());
    ASSERT_EQ(function->output(1).get_node(), function->output("concat").get_node());
    ASSERT_NE(function->output(0).get_node(), function->output("concat").get_node());
}

TEST_P(OVExecNetwork, precisionsAsInOriginalFunction) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));

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
    ASSERT_NO_THROW(execNet = ie->compile_model(m_out_xml_path_1, targetDevice, configuration));

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

}  // namespace test
}  // namespace ov
