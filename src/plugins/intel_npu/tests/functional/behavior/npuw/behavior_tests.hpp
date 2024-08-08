// Copyright (C) 2024 Intel Corporation
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

class BehaviorTestsNPUW : public ::testing::Test {
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

    std::shared_ptr<ov::Model> create_example_model();

private:
    std::vector<std::shared_ptr<void>> m_shared_objects;
};

}  // namespace tests
}  // namespace npuw
}  // namespace ov
