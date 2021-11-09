// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_Negative1) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};   // ignoring end -- slicing to the end

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
}

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_Negative2) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto relu = std::make_shared<ngraph::opset1::Relu>(data);
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};   // ignoring end -- slicing to the end

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
}

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto relu = std::make_shared<ngraph::opset1::Relu>(data);
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};   // ignoring end -- slicing to the end

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto relu = std::make_shared<ngraph::opset1::Relu>(data);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_SkipUselessDeletionRevertCase) {
    {
        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-6, -7, -8, -9});
        auto stride = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1});

        std::vector<int64_t> begin_mask = {1, 1, 1, 1};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset3::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);
        auto relu = std::make_shared<ngraph::opset3::Relu>(ss);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-6, -7, -8, -9});
        auto stride = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1});

        std::vector<int64_t> begin_mask = {1, 1, 1, 1};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset3::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);
        auto relu = std::make_shared<ngraph::opset3::Relu>(ss);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Usefull_Test) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Shared_Test) {
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});

        auto begin1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<ngraph::opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<ngraph::opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ss1, ss2}, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});

        auto begin1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<ngraph::opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ss1, ss1}, 0);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_NotShared_Test) {
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 6, 5, 5});

        auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto split = std::make_shared<ngraph::opset1::Split>(source, axis, 2);

        auto begin1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<ngraph::opset1::StridedSlice>(split->output(0), begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<ngraph::opset1::StridedSlice>(split->output(1), begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ss1, ss2}, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 6, 5, 5});

        auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto split = std::make_shared<ngraph::opset1::Split>(source, axis, 2);

        auto begin1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<ngraph::opset1::StridedSlice>(split->output(0), begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<ngraph::opset1::StridedSlice>(split->output(1), begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ss1, ss2}, 0);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Groupped_Test) {
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});

        auto begin1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {5, 3, 5, 5});
        auto stride1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<ngraph::opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 0, 0});
        auto end2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {5, 5, 5, 5});
        auto stride2 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<ngraph::opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ss1, ss2}, 1);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 5, 5, 5});

        auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto split_sizes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 2});
        auto variadic_split = std::make_shared<ngraph::opset1::VariadicSplit>(source, axis, split_sizes);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{variadic_split->output(0), variadic_split->output(1)}, 1);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{source});
    }
}
