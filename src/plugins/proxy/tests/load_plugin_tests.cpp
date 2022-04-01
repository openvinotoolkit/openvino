// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, get_available_devices) {
    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(ProxyTests, load_and_infer_on_device_without_split_on_default_device) {
    // Model has only add (+ 1) op and reshape
    auto model = create_model_with_reshape();
    auto infer_request = core.compile_model(model, "MOCK").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_size(), output_tensor.get_size());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_NE(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
    // Change input tensor
    {
        auto* data = input_tensor.data<int64_t>();
        for (size_t i = 0; i < input_tensor.get_size(); i++)
            data[i] += 1;
    }
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyTests, load_and_infer_on_device_without_split) {
    auto model = create_model_with_subtract();
    auto infer_request = core.compile_model(model, "MOCK.3").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyTests, load_on_unsupported_plugin) {
    auto model = create_model_with_subtract();
    EXPECT_THROW(core.compile_model(model, "MOCK.0"), ov::Exception);
}

#ifdef HETERO_ENABLED
TEST_F(ProxyTests, load_on_support_with_hetero_plugin) {
    auto model = create_model_with_subtract();
    auto infer_request = core.compile_model(model, "MOCK.1").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}
#endif
