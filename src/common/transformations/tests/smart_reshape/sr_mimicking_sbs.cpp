// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace ov;

TEST(SmartReshapeTests, MimickingSBS) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {6, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, MimickingSBS_1) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {1, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, MimickingSBS_2) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2, 3, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input, opset5::Constant::create(element::i64, {2}, {12, -1}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 1));
}
