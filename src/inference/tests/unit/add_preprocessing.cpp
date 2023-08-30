// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "dev/preprocessing/preprocessing.hpp"
#include "openvino/op/add.hpp"
#include "openvino/runtime/common.hpp"

using namespace testing;

using AddPreprocessingTests = ::testing::Test;

TEST_F(AddPreprocessingTests, AddPreprocessingNCDHW) {
    auto shape = ov::Shape{3, 2, 3, 1, 2};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input, constant);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::AddPreprocessing>();
    ASSERT_NO_THROW(manager.run_passes(model));
}

TEST_F(AddPreprocessingTests, AddPreprocessingBlocked) {
    auto shape = ov::Shape{3, 2, 3, 1, 2, 3};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input, constant);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::AddPreprocessing>();
    ASSERT_NO_THROW(manager.run_passes(model));
}

TEST_F(AddPreprocessingTests, AddPreprocessingScalar) {
    auto shape = ov::Shape{};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, {1});
    auto add = std::make_shared<ov::op::v1::Add>(input, constant);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::AddPreprocessing>();
    ASSERT_NO_THROW(manager.run_passes(model));
}
