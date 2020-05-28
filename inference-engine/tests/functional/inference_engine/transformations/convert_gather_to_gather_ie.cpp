// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/gather_ie.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertGatherToGatherIEStatic1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertGatherToGatherIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto gather = std::make_shared<op::GatherIE>(input, indices, 1);

        f_ref = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGatherToGatherIEStatic2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertGatherToGatherIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze = std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        f_ref = std::make_shared<Function>(NodeVector{squeeze}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGatherToGatherIEDynamic1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertGatherToGatherIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto gather = std::make_shared<op::GatherIE>(input, indices, 1);

        f_ref = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGatherToGatherIEDynamic2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertGatherToGatherIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto unsqueeze = std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<op::GatherIE>(input, unsqueeze, 1);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        f_ref = std::make_shared<Function>(NodeVector{squeeze}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}