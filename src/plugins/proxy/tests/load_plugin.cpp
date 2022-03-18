// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"

class ProxyPluginTests : public ::testing::Test {
public:
    ov::Core core;
    void SetUp() override {
        // TODO: Remove temp plugins from core
        // core.register_plugin(std::string("mock_abc_plugin") + IE_BUILD_POSTFIX, "ABC");
        // core.register_plugin(std::string("mock_bde_plugin") + IE_BUILD_POSTFIX, "BDE");
    }

    std::shared_ptr<ov::Model> create_model_with_evaluates() {
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
        auto const_value =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, std::vector<int64_t>(1));
        auto add = std::make_shared<ov::opset8::Add>(param, const_value);
        auto subtract = std::make_shared<ov::opset8::Subtract>(param, const_value);
        auto result = std::make_shared<ov::opset8::Result>(subtract);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    ov::Tensor create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape) {
        switch (type) {
        case ov::element::Type_t::i64:
            return create_tensor<ov::element_type_traits<ov::element::Type_t::i64>::value_type>(type, shape);
        default:
            break;
        }
        throw ov::Exception("Cannot generate tensor. Unsupported element type.");
    }

private:
    template <class T>
    ov::Tensor create_tensor(const ov::element::Type& type, const ov::Shape& shape) {
        ov::Tensor tensor(type, shape);
        T* data = tensor.data<T>();
        for (size_t i = 0; i < tensor.get_size(); i++) {
            data[i] = static_cast<T>(i);
        }
        return tensor;
    }
};

TEST_F(ProxyPluginTests, get_available_devices) {
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

TEST_F(ProxyPluginTests, load_and_infer_on_device_without_split) {
    auto model = create_model_with_evaluates();
    auto infer_request = core.compile_model(model, "MOCK.3").create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}

TEST_F(ProxyPluginTests, load_on_unsupported_plugin) {
    auto model = create_model_with_evaluates();
    EXPECT_THROW(core.compile_model(model, "MOCK.0"), ov::Exception);
}

#ifdef HETERO_ENABLED
TEST_F(ProxyPluginTests, load_on_support_with_hetero_plugin) {
    auto model = create_model_with_evaluates();
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
