// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <common_test_utils/graph_comparator.hpp>


TEST(GraphComparatorTests, AllEnablePositiveCheck) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add}, ngraph::ParameterVector{ input });
        function = ov::clone_model(*function_ref);
    }
    comparator.enable(FunctionsComparator::NAMES)
    .enable(FunctionsComparator::CONST_VALUES)
    .enable(FunctionsComparator::PRECISIONS)
    .enable(FunctionsComparator::ATTRIBUTES)
    .enable(FunctionsComparator::RUNTIME_KEYS)
    .enable(FunctionsComparator::TENSOR_NAMES);

    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckbyDefault) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto input2 = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto add = std::make_shared<ov::opset8::Add>(input, input2);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input, input2 });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {12});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, NamesCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        auto result = std::make_shared<ov::opset8::Result>(add);
        function_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ngraph::ParameterVector{ input });
        function = ov::clone_model(*function_ref);
        result->set_friendly_name("new_name");
    }
    //?
    comparator.enable(FunctionsComparator::NAMES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, ConstCheckWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {12});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, ConstCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {12});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, TensorNamesCheckNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
        function = ov::clone_model(*function_ref);
        add->get_input_tensor(0).set_names({"new_name"});
    }
    comparator.enable(FunctionsComparator::TENSOR_NAMES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, TensorNamesCheckWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{1});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {1}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
        function = ov::clone_model(*function_ref);
        add->get_input_tensor(0).set_names({"new_name"});
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckAttributesNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
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
                                                          { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{ 1, 1 },
                                                              ov::CoordinateDiff{ 0, 0 },
                                                              ov::CoordinateDiff{ 0, 0 },
                                                              ov::Strides{ 1, 1 });
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ conv }, ngraph::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckPrecisionsNegative) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::f32, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::f32, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::PRECISIONS);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckPrecisionsWithoutEnable) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::f32, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::f32, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfo) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        add->get_rt_info()["my_info"] = 42;
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS);
    auto res = comparator.compare(function, function_ref);
    ASSERT_FALSE(res.valid) << res.message;
}

TEST(GraphComparatorTests, CheckRTInfoReverse) {
    FunctionsComparator comparator(FunctionsComparator::no_default());
    std::shared_ptr<ov::Model> function, function_ref;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ngraph::element::i64, ov::Shape{3});
        auto constant = ov::opset8::Constant::create(ngraph::element::i64, {3}, {0});
        auto add = std::make_shared<ov::opset8::Add>(input, constant);
        add->get_rt_info()["my_info"] = 42;
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::RUNTIME_KEYS);
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
