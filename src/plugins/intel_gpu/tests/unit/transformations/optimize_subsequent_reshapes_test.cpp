// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <memory>

#include "openvino/pass/manager.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

#include <plugin/transformations/optimize_subsequent_reshapes.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, OptimizeSubsequentReshapes1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 1, 4096 });
        auto first_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{ 0, 0, 32, 128 });
        auto first_reshape = std::make_shared<ov::op::v1::Reshape>(input, first_reshape_pattern, true);

        auto second_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{ 0, -1 });
        auto second_reshape = std::make_shared<ov::op::v1::Reshape>(first_reshape, second_reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(second_reshape);

        model = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
        manager.register_pass<OptimizeSubsequentReshapes>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 1, 4096 });
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{ 0, 4096 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(reshape);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, OptimizeSubsequentReshapes2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 1, 4096 });
        auto first_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{ 0, 0, 32, 128 });
        auto first_reshape = std::make_shared<ov::op::v1::Reshape>(input, first_reshape_pattern, true);

        auto second_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{ 0, 32, 1, 0 });
        auto second_reshape = std::make_shared<ov::op::v1::Reshape>(first_reshape, second_reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(second_reshape);

        model = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
        manager.register_pass<OptimizeSubsequentReshapes>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 1, 4096 });
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{ 0, 32, 1, 128 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(reshape);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, OptimizeSubsequentReshapes3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 32, 1, 128 });
        auto first_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{ 0, 1, 32, 0 });
        auto first_reshape = std::make_shared<ov::op::v1::Reshape>(input, first_reshape_pattern, true);

        auto second_reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{ 0, -1 });
        auto second_reshape = std::make_shared<ov::op::v1::Reshape>(first_reshape, second_reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(second_reshape);

        model = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
        manager.register_pass<OptimizeSubsequentReshapes>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, 32, 1, 128 });
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{ 0, 4096 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, reshape_pattern, true);
        auto result = std::make_shared<ov::op::v0::Result>(reshape);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ result }, ov::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
