// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/constant_convert_folding.hpp>
#include "snippets/snippets_isa.hpp"
#include "snippets/pass/constant_convert_folding.hpp"
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

void ConstantConvertFoldingTests::run() {
    ASSERT_TRUE(function);
    std::string name;
    manager.register_pass<ngraph::snippets::pass::ConstantConvertFolding>();
}

TEST_F(ConstantConvertFoldingTests, smoke_Snippets_ConstantConvertFolding_oneConvertTruncation) {
    const auto values = std::vector<float>{1, 2, 3, 4};
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::i32, {1, 4, 1, 1}, values);
        const auto convert = std::make_shared<ngraph::snippets::op::ConvertTruncation>(constant, ov::element::f32);
        const auto add = std::make_shared<ov::op::v1::Add>(data, convert);
        function = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        function_ref = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    run();
}

TEST_F(ConstantConvertFoldingTests, smoke_Snippets_ConstantConvertFolding_oneConvertSaturation) {
    const auto values = std::vector<float>{1, 2, 3, 4};
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::i32, {1, 4, 1, 1}, values);
        const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(constant, ov::element::f32);
        const auto add = std::make_shared<ov::op::v1::Add>(data, convert);
        function = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        function_ref = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    run();
}

TEST_F(ConstantConvertFoldingTests, smoke_Snippets_ConstantConvertFolding_twoConvertTruncationSaturation) {
    const auto values = std::vector<float>{1, 2, 3, 4};
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::i32, {1, 4, 1, 1}, values);
        const auto convert0 = std::make_shared<ngraph::snippets::op::ConvertTruncation>(constant, ov::element::i8);
        const auto convert1 = std::make_shared<ngraph::snippets::op::ConvertSaturation>(convert0, ov::element::f32);
        const auto add = std::make_shared<ov::op::v1::Add>(data, convert1);
        function = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        function_ref = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data});
    }
    run();
}

TEST_F(ConstantConvertFoldingTests, smoke_Snippets_ConstantConvertFolding_ConstantWithSeveralConsumers_Saturation) {
    const auto values = std::vector<float>{1, 2, 3, 4};
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::i32, {1, 4, 1, 1}, values);
        const auto convert0 = std::make_shared<ngraph::snippets::op::ConvertSaturation>(constant, ov::element::f32);
        const auto add = std::make_shared<ov::op::v1::Add>(data, convert0);
        const auto convert1 = std::make_shared<ngraph::snippets::op::ConvertSaturation>(constant, ov::element::f32);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(add, convert1);
        function = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant0 = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto add = std::make_shared<ov::op::v1::Add>(data, constant0);
        const auto constant1 = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(add, constant1);
        function_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    run();
}

TEST_F(ConstantConvertFoldingTests, smoke_Snippets_ConstantConvertFolding_ConvertWithSeveralConsumers_Truncation) {
    const auto values = std::vector<float>{1, 2, 3, 4};
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::i32, {1, 4, 1, 1}, values);
        const auto convert = std::make_shared<ngraph::snippets::op::ConvertTruncation>(constant, ov::element::f32);
        const auto add = std::make_shared<ov::op::v1::Add>(data, convert);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(add, convert);
        function = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});
        const auto constant = ngraph::builder::makeConstant(ov::element::f32, {1, 4, 1, 1}, values);
        const auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(add, constant);
        function_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data});
    }
    run();
}


}  // namespace snippets
}  // namespace test
}  // namespace ov
