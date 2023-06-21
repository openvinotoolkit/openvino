// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/opsets/opset11.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertTopK11ToTopK3) {
    {
        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2, 3, 4});
        const auto k = std::make_shared<ov::opset11::Parameter>(ov::element::i8, ov::Shape{});
        const auto topk = std::make_shared<ov::opset11::TopK>(input,
                                                              k,
                                                              -2,
                                                              ov::op::TopKMode::MAX,
                                                              ov::op::TopKSortType::SORT_VALUES,
                                                              ov::element::i64,
                                                              false);
        topk->set_friendly_name("topk11");

        function = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{input, k});
        manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    }

    {
        const auto input = std::make_shared<ov::opset3::Parameter>(ov::element::i32, ov::Shape{2, 3, 4});
        const auto k = std::make_shared<ov::opset3::Parameter>(ov::element::i8, ov::Shape{});
        const auto topk = std::make_shared<ov::opset3::TopK>(input,
                                                             k,
                                                             -2,
                                                             ov::op::TopKMode::MAX,
                                                             ov::op::TopKSortType::SORT_VALUES,
                                                             ov::element::i64);
        topk->set_friendly_name("topk11");

        function_ref = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{input, k});
    }
}

TEST_F(TransformationTestsF, ConvertTopK11ToTopK3StableMode) {
    {
        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2, 3, 4});
        const auto k = std::make_shared<ov::opset11::Parameter>(ov::element::i8, ov::Shape{});
        const auto topk = std::make_shared<ov::opset11::TopK>(input,
                                                              k,
                                                              -2,
                                                              ov::op::TopKMode::MAX,
                                                              ov::op::TopKSortType::SORT_VALUES,
                                                              ov::element::i64,
                                                              true);
        topk->set_friendly_name("topk11");

        function = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{input, k});
        manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    }

    {
        const auto input = std::make_shared<ov::opset3::Parameter>(ov::element::i32, ov::Shape{2, 3, 4});
        const auto k = std::make_shared<ov::opset3::Parameter>(ov::element::i8, ov::Shape{});
        const auto topk = std::make_shared<ov::opset3::TopK>(input,
                                                             k,
                                                             -2,
                                                             ov::op::TopKMode::MAX,
                                                             ov::op::TopKSortType::SORT_VALUES,
                                                             ov::element::i64);
        topk->set_friendly_name("topk11");

        function_ref = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{input, k});
    }
}
