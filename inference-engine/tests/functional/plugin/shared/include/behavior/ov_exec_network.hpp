// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include <transformations/serialize.hpp>
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/runtime.hpp"

namespace ov {
namespace test {
using OVExecNetwork = ov::test::BehaviorTestsBasic;

// Load correct network to Plugin to get executable network
TEST_P(OVExecNetwork, precisionsAsInOriginalFunction) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));

    EXPECT_EQ(function->get_parameters().size(), execNet.get_parameters().size());
    auto ref_parameter = function->get_parameters().back();
    auto actual_parameter = execNet.get_parameters().back();
    EXPECT_EQ(ref_parameter->get_element_type(), actual_parameter->get_element_type());
    EXPECT_EQ(ref_parameter->get_shape(), actual_parameter->get_shape());
    EXPECT_EQ(ref_parameter->get_friendly_name(), actual_parameter->get_friendly_name());

    EXPECT_EQ(function->get_results().size(), execNet.get_results().size());
    auto ref_result = function->get_results().back();
    auto actual_result = execNet.get_results().back();
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

    EXPECT_EQ(function->get_parameters().size(), execNet.get_parameters().size());
    auto ref_parameter = function->get_parameters().back();
    auto actual_parameter = execNet.get_parameters().back();
    EXPECT_EQ(elementType, ref_parameter->get_element_type());
    EXPECT_EQ(ref_parameter->get_element_type(), actual_parameter->get_element_type());
    EXPECT_EQ(ref_parameter->get_shape(), actual_parameter->get_shape());
    EXPECT_EQ(ref_parameter->get_friendly_name(), actual_parameter->get_friendly_name());

    EXPECT_EQ(function->get_results().size(), execNet.get_results().size());
    auto ref_result = function->get_results().back();
    auto actual_result = execNet.get_results().back();
    EXPECT_EQ(ref_result->get_element_type(), actual_result->get_element_type());
    EXPECT_EQ(ref_result->get_shape(), actual_result->get_shape());
    EXPECT_EQ(ref_result->get_friendly_name(), actual_result->get_friendly_name());
}

}  // namespace test
}  // namespace ov
