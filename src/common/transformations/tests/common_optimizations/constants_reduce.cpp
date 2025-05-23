// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/constants_reduce.hpp"

#include <gtest/gtest.h>
#include <math.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, ConstantsReduce) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 4});

    // Intentionally equal to each other
    auto add_constant_1 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_2 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_1 = std::make_shared<opset8::Add>(param, add_constant_1);
    auto add_2 = std::make_shared<opset8::Add>(add_1, add_constant_2);

    auto result = std::make_shared<ov::op::v0::Result>(add_2);
    auto f = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ConstantsReduce>();
    pass_manager.run_passes(f);

    // One constant should be reduced since they are equal
    ASSERT_EQ(count_ops_of_type<opset8::Constant>(f), 1);
}

TEST(TransformationTests, ConstantsReduceChain) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 4});

    // Intentionally equal to each other
    auto add_constant_1 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_2 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_3 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_4 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});

    // Intentionally different
    auto add_constant_5 = opset8::Constant::create(element::f32, Shape{1, 4}, {2.0, 2.0, 3.0, 4.0});
    auto add_1 = std::make_shared<opset8::Add>(param, add_constant_1);
    auto add_2 = std::make_shared<opset8::Add>(add_1, add_constant_2);
    auto add_3 = std::make_shared<opset8::Add>(add_2, add_constant_3);
    auto add_4 = std::make_shared<opset8::Add>(add_3, add_constant_4);
    auto add_5 = std::make_shared<opset8::Add>(add_4, add_constant_5);

    auto result = std::make_shared<ov::op::v0::Result>(add_5);
    auto f = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ConstantsReduce>();
    pass_manager.run_passes(f);

    // All constants should be reduced to one except the one that is different
    ASSERT_EQ(count_ops_of_type<opset8::Constant>(f), 2);
}

TEST(TransformationTests, ConstantsReduceChain2) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 4});

    // Intentionally equal to each other
    auto add_constant_1 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_2 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_3 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_4 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_5 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});

    auto add_1 = std::make_shared<opset8::Add>(param, add_constant_1);
    auto add_2 = std::make_shared<opset8::Add>(add_1, add_constant_2);
    auto add_3 = std::make_shared<opset8::Add>(add_2, add_constant_3);
    auto add_4 = std::make_shared<opset8::Add>(add_3, add_constant_4);
    auto add_5 = std::make_shared<opset8::Add>(add_4, add_constant_5);

    auto result = std::make_shared<ov::op::v0::Result>(add_5);
    auto f = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ConstantsReduce>();
    pass_manager.run_passes(f);

    // All constants should be reduced to one
    ASSERT_EQ(count_ops_of_type<opset8::Constant>(f), 1);
}

TEST(TransformationTests, ConstantsReduceNeg) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 4});

    // Intentionally unequal to each other
    auto add_constant_1 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.0});
    auto add_constant_2 = opset8::Constant::create(element::f32, Shape{1, 4}, {1.0, 2.0, 3.0, 4.5});
    auto add_1 = std::make_shared<opset8::Add>(param, add_constant_1);
    auto add_2 = std::make_shared<opset8::Add>(add_1, add_constant_2);

    auto result = std::make_shared<ov::op::v0::Result>(add_2);
    auto f = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ConstantsReduce>();
    pass_manager.run_passes(f);

    // No reduction here
    ASSERT_EQ(count_ops_of_type<opset8::Constant>(f), 2);
}
