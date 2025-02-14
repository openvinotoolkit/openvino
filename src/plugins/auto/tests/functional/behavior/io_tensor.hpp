// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <thread>

#include "auto_func_test.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace auto_plugin {
namespace tests {

using test_params = std::tuple<std::string, ov::AnyMap>;

class InferRequest_IOTensor_Test : public AutoFuncTests, public ::testing::WithParamInterface<test_params> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<test_params> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        if (!configuration.empty()) {
            for (auto& iter : configuration) {
                result << "priority=" << iter.first << "_" << iter.second.as<std::string>();
            }
        }
        return result.str();
    }

    void SetUp() override;
    void TearDown() override;

protected:
    std::string target_device;
    ov::InferRequest req;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
    ov::AnyMap property;
};
}  // namespace tests
}  // namespace auto_plugin
}  // namespace ov