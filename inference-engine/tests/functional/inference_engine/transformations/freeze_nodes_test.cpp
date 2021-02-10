// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/freeze_nodes.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, FreezeParameter) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto in_a = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto in_b = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_b);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{in_a, in_b});

        using data = std::vector<std::vector<char>>;
        std::vector<data> values(1, data(1, std::vector<char>(2 * sizeof(float))));
        auto* fp_val = reinterpret_cast<float*>(values[0][0].data());

        fp_val[0] = 3;
        fp_val[1] = 5;

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::FreezeNodes>(ngraph::NodeVector{in_b}, values);
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in_a = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto in_b = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{3, 5});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_b);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{in_a});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_parameters().size(), 1);
}

TEST(TransformationTests, FreezeAllParameters) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto in_a = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto in_b = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_b);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{in_a, in_b});

        using data = std::vector<std::vector<char>>;
        std::vector<data> values(2, data(1, std::vector<char>(2 * sizeof(float))));
        auto* fp_val_a = reinterpret_cast<float*>(values[0][0].data());
        auto* fp_val_b = reinterpret_cast<float*>(values[1][0].data());

        fp_val_a[0] = 3;
        fp_val_a[1] = 5;

        fp_val_b[0] = 2;
        fp_val_b[1] = 4;

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::FreezeNodes>(ngraph::NodeVector{in_b, in_a}, values);
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in_a = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{2, 4});
        auto in_b = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{3, 5});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_b);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_parameters().size(), 0);
}

TEST(TransformationTests, FreezeParameterOneOutputToMany) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto in_a = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_a);
        auto add_2 = std::make_shared<ngraph::opset6::Add>(add, in_a);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add_2}, ngraph::ParameterVector{in_a});

        using data = std::vector<std::vector<char>>;
        std::vector<data> values(1, data(1, std::vector<char>(2 * sizeof(float))));
        auto* fp_val_a = reinterpret_cast<float*>(values[0][0].data());

        fp_val_a[0] = 3;
        fp_val_a[1] = 5;

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::FreezeNodes>(ngraph::NodeVector{in_a}, values);
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in_a = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{3, 5});
        auto add = std::make_shared<ngraph::opset6::Add>(in_a, in_a);
        auto add_2 = std::make_shared<ngraph::opset6::Add>(add, in_a);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add_2}, ngraph::ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_parameters().size(), 0);
}


TEST(TransformationTests, FreezeNodeWithMultipleOutputs) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto in_a = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2});
        auto axis = std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64, ngraph::Shape{}, 1);
        auto split = std::make_shared<ngraph::opset6::Split>(in_a, axis, 2);

        f = std::make_shared<ngraph::Function>(ngraph::OutputVector {split->output(0), split->output(1)}, ngraph::ParameterVector{in_a});

        using data = std::vector<std::vector<char>>;
        std::vector<data> values(1, data(2, std::vector<char>(3 * sizeof(float))));
        auto* fp_val_a = reinterpret_cast<float*>(values[0][0].data());
        auto* fp_val_b = reinterpret_cast<float*>(values[0][1].data());

        fp_val_a[0] = 3;
        fp_val_a[1] = 5;
        fp_val_a[2] = 7;

        fp_val_b[0] = 2;
        fp_val_b[1] = 4;
        fp_val_b[2] = 6;

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::FreezeNodes>(ngraph::NodeVector{split}, values);
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in_a = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{3, 1}, std::vector<float>{3, 5, 7});
        auto in_b = std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, ngraph::Shape{3, 1}, std::vector<float>{2, 4, 6});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{in_a, in_b}, ngraph::ParameterVector{});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_parameters().size(), 1);
}
