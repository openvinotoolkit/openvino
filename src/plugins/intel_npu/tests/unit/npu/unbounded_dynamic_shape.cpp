// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "model_validation.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

// Unit tests for intel_npu::validate_no_unbounded_dynamic_dimensions(), the guard that rejects models with unbounded
// dynamic dimensions (upper bound == INT64_MAX) before they reach the NPU compiler. The guard is exercised directly,
// without compiling a model for a device, so these tests run on every CI runner regardless of NPU availability.

using intel_npu::validate_no_unbounded_dynamic_dimensions;

namespace {

std::shared_ptr<ov::Model> make_single_io_model(const ov::PartialShape& input_shape, const std::string& input_name) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    parameter->set_friendly_name(input_name);
    auto result = std::make_shared<ov::op::v0::Result>(parameter);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
}

}  // namespace

TEST(NPUUnboundedDynamicShape, RejectsUnboundedDynamicInput) {
    // ov::Dimension::dynamic() spans [0, INT64_MAX]: dynamic with no finite upper bound.
    const auto model = make_single_io_model({1, ov::Dimension::dynamic(), 64}, "test_input");

    try {
        validate_no_unbounded_dynamic_dimensions(model);
        FAIL() << "Expected validate_no_unbounded_dynamic_dimensions() to throw for an unbounded dynamic dimension";
    } catch (const ov::Exception& e) {
        const std::string message = e.what();
        EXPECT_THAT(message, ::testing::HasSubstr("unbounded dynamic dimensions"));
        EXPECT_THAT(message, ::testing::HasSubstr("test_input"));
        EXPECT_THAT(message, ::testing::HasSubstr("model.reshape"));
    }
}

TEST(NPUUnboundedDynamicShape, AllowsBoundedDynamicInput) {
    // A finite upper bound (1..512) must not be treated as unbounded.
    const auto model = make_single_io_model({1, ov::Dimension(1, 512), 64}, "test_input_bounded");
    EXPECT_NO_THROW(validate_no_unbounded_dynamic_dimensions(model));
}

TEST(NPUUnboundedDynamicShape, AllowsFullyStaticModel) {
    const auto model = make_single_io_model({1, 3, 64}, "test_input_static");
    EXPECT_NO_THROW(validate_no_unbounded_dynamic_dimensions(model));
}
