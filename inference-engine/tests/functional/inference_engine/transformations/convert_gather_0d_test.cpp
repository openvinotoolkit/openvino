// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/op_conversions/convert_gather_0d.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertGather0DStatic1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertGather0D>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{15, 4, 20, 28});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f_ref = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGather0DStatic2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto gather = std::make_shared<opset1::Gather>(input, indices, axis_const);

        f = std::make_shared<Function>(NodeVector{gather}, ParameterVector{input, indices});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertGather0D>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{6, 12, 10, 24});
        auto indices = std::make_shared<opset1::Parameter>(element::f32, Shape{});
        auto axis_const = opset1::Constant::create(element::i64, Shape{}, {1});
        auto unsqueeze = std::make_shared<opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
        auto gather = std::make_shared<opset1::Gather>(input, unsqueeze, axis_const);
        auto squeeze = std::make_shared<opset1::Squeeze>(gather, opset1::Constant::create(element::i64, Shape{1}, {1}));

        f_ref = std::make_shared<Function>(NodeVector{squeeze}, ParameterVector{input, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
