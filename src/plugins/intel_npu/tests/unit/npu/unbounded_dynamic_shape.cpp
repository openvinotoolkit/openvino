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
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"

// Unit tests for intel_npu::validate_no_unbounded_dynamic_dimensions(), the guard that rejects models with unbounded
// dynamic dimensions (upper bound == INT64_MAX) before they reach the NPU compiler. The guard is exercised directly,
// without compiling a model for a device, so these tests run on every CI runner regardless of NPU availability.

using intel_npu::validate_no_unbounded_dynamic_dimensions;
using ::testing::HasSubstr;
using ::testing::Not;

namespace {

std::shared_ptr<ov::op::v0::Parameter> make_param(const ov::PartialShape& shape, const std::string& name) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name(name);
    return parameter;
}

std::shared_ptr<ov::Model> make_single_io_model(const ov::PartialShape& input_shape, const std::string& input_name) {
    auto parameter = make_param(input_shape, input_name);
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
        EXPECT_THAT(message, HasSubstr("unbounded dynamic dimensions"));
        EXPECT_THAT(message, HasSubstr("Parameter"));
        EXPECT_THAT(message, HasSubstr("test_input"));
        EXPECT_THAT(message, HasSubstr("[1]"));  // the offending dimension index
        EXPECT_THAT(message, HasSubstr("model.reshape"));
    }
}

TEST(NPUUnboundedDynamicShape, ReportsOffendingParameterInMultiInputModel) {
    // Only the second parameter is unbounded (at index 2); the message must name it and its index, not the valid one.
    auto static_param = make_param({4, 8, 2}, "static_input");
    auto unbounded_param = make_param({4, 8, ov::Dimension::dynamic()}, "unbounded_input");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(static_param),
                                                              std::make_shared<ov::op::v0::Result>(unbounded_param)},
                                             ov::ParameterVector{static_param, unbounded_param});

    try {
        validate_no_unbounded_dynamic_dimensions(model);
        FAIL() << "Expected a throw for the unbounded parameter";
    } catch (const ov::Exception& e) {
        const std::string message = e.what();
        EXPECT_THAT(message, HasSubstr("unbounded_input"));
        EXPECT_THAT(message, HasSubstr("[2]"));
        EXPECT_THAT(message, Not(HasSubstr("static_input")));
    }
}

TEST(NPUUnboundedDynamicShape, RejectsUnboundedDynamicOutput) {
    // Static scalar inputs feeding Range produce a fully unbounded 1-D output, so the parameter loop passes and the
    // guard must fire on the result/output branch, naming the producing node (Result nodes have no friendly name).
    auto start = make_param({}, "start");
    auto stop = make_param({}, "stop");
    auto step = make_param({}, "step");
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
    range->set_friendly_name("range_node");
    auto result = std::make_shared<ov::op::v0::Result>(range);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{start, stop, step});

    try {
        validate_no_unbounded_dynamic_dimensions(model);
        FAIL() << "Expected a throw for the unbounded output";
    } catch (const ov::Exception& e) {
        const std::string message = e.what();
        EXPECT_THAT(message, HasSubstr("unbounded dynamic dimensions"));
        EXPECT_THAT(message, HasSubstr("Output"));
        EXPECT_THAT(message, HasSubstr("range_node"));
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

TEST(NPUUnboundedDynamicShape, AllowsDynamicRankParameter) {
    // A fully dynamic rank has no indexable dimensions; the guard must skip it (rejected elsewhere in the pipeline).
    const auto model = make_single_io_model(ov::PartialShape::dynamic(), "dynamic_rank_input");
    EXPECT_NO_THROW(validate_no_unbounded_dynamic_dimensions(model));
}
