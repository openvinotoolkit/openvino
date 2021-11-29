// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, SplitSqueezeConcatFusion) {
    size_t num_splits = 4;

    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SplitSqueezeConcatFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 3, 4, 2, 5 });
        auto transpose = std::make_shared<ngraph::opset7::Transpose>(input, transpose_order);
        auto reshape_shape = ngraph::opset7::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{ 5 },
                                                                       { 1, 2, 640, 20, 2 * (int64_t)num_splits });
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(transpose, reshape_shape, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SplitSqueezeConcatFusionNegativeCaseNotAllSplitOutputsGoToSqueeze) {
    size_t num_splits = 4;

    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits - 1);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SplitSqueezeConcatFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits - 1);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SplitSqueezeConcatFusionNegativeCaseSplitOutputsGoInDifferentOrder) {
    size_t num_splits = 4;

    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        std::swap(squeeze_vec[1], squeeze_vec[2]);

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SplitSqueezeConcatFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        std::swap(squeeze_vec[1], squeeze_vec[2]);

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SplitSqueezeConcatFusionNegativeCaseSplitAxisDifferentFromSqueezeAxis) {
    size_t num_splits = 4;

    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 0 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SplitSqueezeConcatFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, num_splits, 640, 20, 2 });
        auto split_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 2 });
        auto split = std::make_shared<ngraph::opset7::Split>(input, split_axis, num_splits);
        ngraph::OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 0 });
            squeeze_vec[i] = std::make_shared<ngraph::opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(squeeze_vec, 4);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
