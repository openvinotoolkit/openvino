// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <common_test_utils/graph_comparator.hpp>

TEST(GraphComparatorTests, PositiveCheck) {
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add}, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add}, ngraph::ParameterVector{ input });
    }
    ASSERT_NO_THROW(accuracy_check(function_ref, function));
}

TEST(GraphComparatorTests, NegativeCheck) {
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {12});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add}, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {200});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add}, ngraph::ParameterVector{ input });
    }
    ASSERT_ANY_THROW(accuracy_check(function_ref, function));
}

TEST(GraphComparatorTests, CheckConvNegative) {
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f16,
                                                          ov::Shape{ 1, 3, 3, 3 },
                                                          { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{ 1, 1 },
                                                              ov::CoordinateDiff{ 1, 1 },
                                                              ov::CoordinateDiff{ 1, 1 },
                                                              ov::Strides{ 1, 1 });
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f16,
                                                          ov::Shape{ 1, 3, 3, 3 },
                                                          { 1, 9, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 9, 5, 6, 7, 8, 9, 1, 2, 3, 4, 9, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{ 1, 1 },
                                                              ov::CoordinateDiff{ 1, 1 },
                                                              ov::CoordinateDiff{ 1, 1 },
                                                              ov::Strides{ 1, 1 });
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ input });
    }
    ASSERT_ANY_THROW(accuracy_check(function_ref, function));
}

