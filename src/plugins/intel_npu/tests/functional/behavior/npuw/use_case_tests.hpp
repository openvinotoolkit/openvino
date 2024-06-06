// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"

#include "mocks/mock_plugins.hpp"
#include "mocks/register_in_ov.hpp"

namespace ov {
namespace npuw {
namespace tests {

class NPUWUseCaseTests : public ::testing::Test {
public:
    ov::Core core;
    std::shared_ptr<MockNpuPlugin> npu_plugin;
    std::shared_ptr<MockCpuPlugin> cpu_plugin;
    std::shared_ptr<ov::Model> model;


    void SetUp() override {
        model = create_example_model();
        npu_plugin = std::make_shared<MockNpuPlugin>();
        npu_plugin->create_implementation();
        cpu_plugin = std::make_shared<MockCpuPlugin>();
        cpu_plugin->create_implementation();
    }

    // Make sure it is called after expectations are set!
    void register_mock_plugins_in_ov() {
        m_shared_objects.push_back(reg_plugin<MockNpuPlugin>(core, npu_plugin));
        m_shared_objects.push_back(reg_plugin<MockCpuPlugin>(core, cpu_plugin));
    }

    ov::TensorVector create_tensors(std::vector<ov::Output<const ov::Node>> data) {
        ov::TensorVector tensors;
        for (const auto& el : data) {
            const ov::Shape& shape = el.get_shape();
            const ov::element::Type& precision = el.get_element_type();
            tensors.emplace_back(precision, shape);
        }
        return tensors;
    }

    std::shared_ptr<ov::Model> create_example_model();

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const ov::element::Type& element_type,
                              const ov::Shape& shape) {
        if (!tensor || tensor->get_element_type() != element_type) {
            tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(element_type, shape), nullptr);
        } else {
            tensor->set_shape(shape);
        }
    }

    std::vector<std::shared_ptr<void>> m_shared_objects;
};

}  // namespace tests
}  // namespace npuw
}  // namespace ov
