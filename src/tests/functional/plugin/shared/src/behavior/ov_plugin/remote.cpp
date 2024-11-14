// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"
#include "transformations/utils/utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

namespace ov {
namespace test {

std::string OVRemoteTest::getTestCaseName(testing::TestParamInfo<RemoteTensorParams> obj) {
    ov::element::Type element_type;
    std::string target_device;
    ov::AnyMap config;
    std::pair<ov::AnyMap, ov::AnyMap> param_pair;
    std::tie(element_type, target_device, config, param_pair) = obj.param;
    ov::AnyMap context_parameters;
    ov::AnyMap tensor_parameters;
    std::tie(context_parameters, tensor_parameters) = param_pair;
    std::ostringstream result;
    result << "element_type=" << element_type;
    result << "targetDevice=" << target_device;
    for (auto& configItem : config) {
        result << "configItem=" << configItem.first << "_";
        configItem.second.print(result);
        result << "_";
    }
    result << "__context_parameters=";
    for (auto& param : context_parameters) {
        result << param.first << "_";
        PrintTo(param.second, &result);
    }
    result << "__tensor_parameters=";
    for (auto& param : tensor_parameters) {
        result << param.first << "_";
        PrintTo(param.second, &result);
    }
    return result.str();
}

void OVRemoteTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::pair<ov::AnyMap, ov::AnyMap> param_pair;
    std::tie(element_type, target_device, config, param_pair) = GetParam();
    std::tie(context_parameters, tensor_parameters) = param_pair;
    function = ov::test::utils::make_conv_pool_relu({1, 1, 32, 32}, element_type);
    exec_network = core.compile_model(function, target_device, config);
    infer_request = exec_network.create_infer_request();
    input = function->get_parameters().front();
}

void OVRemoteTest::TearDown() {
    infer_request = {};
    exec_network = {};
}

TEST_P(OVRemoteTest, canCreateRemote) {
    auto context = context_parameters.empty()
        ? core.get_default_context(target_device)
        : core.create_context(target_device, context_parameters);

    ov::AnyMap params;
    std::string device;

    OV_ASSERT_NO_THROW(params = context.get_params());
    OV_ASSERT_NO_THROW(device = context.get_device_name());
    for (auto&& param : context_parameters) {
        ASSERT_NE(params.find(param.first), params.end());
    }
    ASSERT_EQ(target_device, device);
    ov::RemoteTensor remote_tensor;
    OV_ASSERT_NO_THROW(remote_tensor = context.create_tensor(input->get_element_type(), input->get_shape(), tensor_parameters));

    OV_ASSERT_NO_THROW(params = remote_tensor.get_params());
    OV_ASSERT_NO_THROW(device = remote_tensor.get_device_name());
    for (auto&& param : tensor_parameters) {
        ASSERT_NE(params.find(param.first), params.end());
    }
    ASSERT_EQ(target_device, device);
}

TEST_P(OVRemoteTest, remoteTensorAsTensor) {
    auto context = context_parameters.empty()
        ? core.get_default_context(target_device)
        : core.create_context(target_device, context_parameters);

    auto remote_tensor = context.create_tensor(input->get_element_type(), input->get_shape(), tensor_parameters);

    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = remote_tensor);
    ASSERT_THROW(tensor.data(), ov::Exception);
    OV_ASSERT_NO_THROW(tensor.get_element_type());
    ASSERT_EQ(input->get_element_type(), tensor.get_element_type());
    OV_ASSERT_NO_THROW(tensor.get_shape());
    ASSERT_EQ(input->get_shape(), tensor.get_shape());
}

TEST_P(OVRemoteTest, inferWithRemoteNoThrow) {
    auto context = context_parameters.empty()
        ? core.get_default_context(target_device)
        : core.create_context(target_device, context_parameters);

    {
        auto input_remote_tensor = context.create_tensor(input->get_element_type(), input->get_shape(), tensor_parameters);
        OV_ASSERT_NO_THROW(infer_request.set_input_tensor(0, input_remote_tensor));
        OV_ASSERT_NO_THROW(infer_request.infer());
    }
    auto output = function->get_results().front();
    {// Host accessable output if input is remote by default
        ov::Tensor tensor;
        OV_ASSERT_NO_THROW(tensor = infer_request.get_output_tensor(0));
        OV_ASSERT_NO_THROW(tensor.data());
    }
    {// Infer with remote on input and outputs
        auto output_remote_tensor = context.create_tensor(output->get_element_type(), output->get_shape(), tensor_parameters);
        OV_ASSERT_NO_THROW(infer_request.set_output_tensor(0, output_remote_tensor));
        OV_ASSERT_NO_THROW(infer_request.infer());
    }
}

}  // namespace test
}  // namespace ov
