// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace proxy {
namespace tests {

class ProxyTests : public ::testing::Test {
public:
    ov::Core core;
    void SetUp() override {
        // TODO: Remove temp plugins from core
        // core.register_plugin(std::string("mock_abc_plugin") + IE_BUILD_POSTFIX, "ABC");
        // core.register_plugin(std::string("mock_bde_plugin") + IE_BUILD_POSTFIX, "BDE");
    }

    std::shared_ptr<ov::Model> create_model_with_subtract() {
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::opset8::Add>(param, const_value);
        add->set_friendly_name("add");
        auto subtract = std::make_shared<ov::opset8::Subtract>(add, const_value);
        subtract->set_friendly_name("sub");
        auto result = std::make_shared<ov::opset8::Result>(subtract);
        result->set_friendly_name("res");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    std::shared_ptr<ov::Model> create_model_with_subtract_reshape() {
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::opset8::Add>(param, const_value);
        add->set_friendly_name("add");
        auto subtract = std::make_shared<ov::opset8::Subtract>(add, const_value);
        subtract->set_friendly_name("sub");
        auto reshape_val = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        reshape_val->set_friendly_name("reshape_val");
        auto reshape = std::make_shared<ov::opset8::Reshape>(subtract, reshape_val, true);
        reshape->set_friendly_name("reshape");
        auto result = std::make_shared<ov::opset8::Result>(reshape);
        result->set_friendly_name("res");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    std::shared_ptr<ov::Model> create_model_with_subtract_reshape_relu() {
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
        param->set_friendly_name("input");
        auto const_value = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto add = std::make_shared<ov::opset8::Add>(param, const_value);
        add->set_friendly_name("add");
        auto subtract = std::make_shared<ov::opset8::Subtract>(add, const_value);
        subtract->set_friendly_name("sub");
        auto reshape_val = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        reshape_val->set_friendly_name("reshape_val");
        auto reshape = std::make_shared<ov::opset8::Reshape>(subtract, reshape_val, true);
        reshape->set_friendly_name("reshape");
        auto relu = std::make_shared<ov::opset8::Relu>(reshape);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ov::opset8::Result>(relu);
        result->set_friendly_name("res");
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

}  // namespace tests
}  // namespace proxy
}  // namespace ov
