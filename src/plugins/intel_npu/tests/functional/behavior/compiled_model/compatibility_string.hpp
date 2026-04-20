// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>

#include "common/functions.hpp"
#include "common/utils.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class CompatibilityStringTest : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> create_dummy_model() {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 224, 224});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "DummyModel");
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
