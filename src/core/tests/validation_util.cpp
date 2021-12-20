// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"

using namespace ov;

template <typename T, typename U>
static void test_evaluate_subgraph(const Output<Node>& subgraph,
                                   const std::vector<T>& input,
                                   const std::vector<U>& expected) {
    auto ret = evaluate_subgraph(subgraph);
    ASSERT_NE(ret, nullptr);
    auto actual = ret->template cast_vector<U>();
    ASSERT_EQ(expected, actual);
}

template <typename T, typename U>
static void test_evaluate_subgraph_with_convert_subtract(const element::Type_t& src_type,
                                                         const element::Type_t& dst_type,
                                                         const std::vector<T>& input) {
    auto constant = opset8::Constant::create(src_type, Shape{input.size()}, input);
    auto convert = std::make_shared<opset8::Convert>(constant, dst_type);
    auto two = opset8::Constant::create(dst_type, Shape{1}, {2});
    auto sub = std::make_shared<opset8::Subtract>(convert, two);
    std::vector<U> expected;
    expected.reserve(input.size());
    for (auto x : input) {
        expected.push_back(static_cast<U>(x) - 2);
    }
    test_evaluate_subgraph(sub, input, expected);
}

TEST(evaluate_subgraph, convert_subtract_subgraph) {
    {
        std::vector<int8_t> input{-128, -1, 0, 1, 127};
        test_evaluate_subgraph_with_convert_subtract<int8_t, float>(element::i8, element::f32, input);
    }
    {
        std::vector<uint8_t> input{0, 1, 255};
        test_evaluate_subgraph_with_convert_subtract<uint8_t, float>(element::u8, element::f32, input);
    }
    {
        std::vector<int8_t> input{-8, -1, 0, 1, 7};
        test_evaluate_subgraph_with_convert_subtract<int8_t, float>(element::i4, element::f32, input);
    }
    {
        std::vector<uint8_t> input{0, 1, 15};
        test_evaluate_subgraph_with_convert_subtract<uint8_t, float>(element::u4, element::f32, input);
    }
}

TEST(evaluate_subgraph, split) {
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto constant = opset8::Constant::create(element::f32, Shape{input.size()}, input);
    auto mul = std::make_shared<opset8::Multiply>(constant, opset8::Constant::create(element::f32, Shape{}, {1}));
    auto shape = std::make_shared<opset8::ShapeOf>(mul);
    auto len_0 = std::make_shared<opset8::Divide>(shape, opset8::Constant::create(element::i64, Shape{}, {2}));
    auto len_1 = std::make_shared<opset8::Subtract>(shape, len_0);
    auto lenghts = std::make_shared<opset8::Concat>(OutputVector{len_0, len_1}, 0);
    auto axis = opset8::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<opset8::VariadicSplit>(mul, axis, lenghts);
    auto split_outputs = split->outputs();
    std::vector<float> expected(std::next(input.begin(), input.size() / 2), input.end());
    test_evaluate_subgraph(split_outputs[1], input, expected);
}
