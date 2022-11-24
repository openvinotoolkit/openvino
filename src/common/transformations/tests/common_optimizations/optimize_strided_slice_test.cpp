// Copyright (C) 2018-2022 Intel Corporation
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
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/core/partial_shape.hpp"

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

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_use_shapes_false) {
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
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    // No UselessStridedSliceEraser transformation if use_shapes == false
}

TEST_F(TransformationTestsF, OptimizeSS_Shared_Test_use_shapes_false) {
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
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    // No SharedStridedSliceEraser transformation if use_shapes == false
}

TEST_F(TransformationTestsF, OptimizeSS_Groupped_Test_use_shapes_false) {
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
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    // No GroupedStridedSliceOptimizer transformation if use_shapes == false
}

TEST_F(TransformationTestsF, SliceToStridedSlice_default_axes) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_axes_const_sorted_full) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 2, 3});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_all_const) {
    {
        auto data = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{4}, {2, 3, 4, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        auto step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{4}, {2, 3, 4, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        auto stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_all_const_fold) {
    {
        auto data = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{4}, {2, 3, 4, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        auto step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto sliced_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {3, 4});
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sliced_const}, ngraph::ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_sorted_less) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 2});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto start = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto stop = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 2});
        auto zero = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        const auto default_begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0});
        const auto begin = std::make_shared<ngraph::opset8::ScatterUpdate>(default_begin,
                                                                           axes,
                                                                           start,
                                                                           zero);

        const auto default_end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0});
        const auto end = std::make_shared<ngraph::opset8::ScatterUpdate>(default_end,
                                                                         axes,
                                                                         stop,
                                                                         zero);

        const auto default_stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1});
        const auto stride = std::make_shared<ngraph::opset8::ScatterUpdate>(default_stride,
                                                                            axes,
                                                                            step,
                                                                            zero);
        std::vector<int64_t> begin_end_mask = {1, 0, 0};
        auto strided_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_unsorted) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto start = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto stop = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto zero = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});
        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 1});

        const auto default_begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto begin = std::make_shared<ngraph::opset8::ScatterUpdate>(default_begin,
                                                                           axes,
                                                                           start,
                                                                           zero);

        const auto default_end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto end = std::make_shared<ngraph::opset8::ScatterUpdate>(default_end,
                                                                         axes,
                                                                         stop,
                                                                         zero);

        const auto default_stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        const auto stride = std::make_shared<ngraph::opset8::ScatterUpdate>(default_stride,
                                                                            axes,
                                                                            step,
                                                                            zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_negative_sorted) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, -3, 2, -1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto stride = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, begin, end, stride});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_dyn_shape_axes_const_negative_unsorted) {
    {
        auto data_shape = ov::PartialShape{ov::Dimension(-1), ov::Dimension(2, 6), 4, ov::Dimension(-1)};
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});

        auto axes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, -3});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data_shape = ov::PartialShape{ov::Dimension(-1), ov::Dimension(2, 6), 4, ov::Dimension(-1)};
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, data_shape);
        auto start = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});
        auto stop = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::PartialShape{ov::Dimension(-1)});

        auto zero = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});
        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 1});

        const auto default_begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto begin = std::make_shared<ngraph::opset8::ScatterUpdate>(default_begin,
                                                                           axes,
                                                                           start,
                                                                           zero);

        const auto default_end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto end = std::make_shared<ngraph::opset8::ScatterUpdate>(default_end,
                                                                         axes,
                                                                         stop,
                                                                         zero);

        const auto default_stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        const auto stride = std::make_shared<ngraph::opset8::ScatterUpdate>(default_stride,
                                                                            axes,
                                                                            step,
                                                                            zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_static_shape_axes_const_negative_unsorted) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto axes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, -3});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto start = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto stop = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        auto zero = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});
        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 1});

        const auto default_begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto begin = std::make_shared<ngraph::opset8::ScatterUpdate>(default_begin,
                                                                           axes,
                                                                           start,
                                                                           zero);

        const auto default_end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        const auto end = std::make_shared<ngraph::opset8::ScatterUpdate>(default_end,
                                                                         axes,
                                                                         stop,
                                                                         zero);

        const auto default_stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});
        const auto stride = std::make_shared<ngraph::opset8::ScatterUpdate>(default_stride,
                                                                            axes,
                                                                            step,
                                                                            zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_dyn_rank_axes_const_positive) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 2, 3});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto stride = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data, begin, end, stride});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_dyn_rank_axes_const_negative) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto end = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});
        auto step = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto axes = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, -3, 2, -1});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin, end, step});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    // No transformation for negative axes and dynamic data rank
}

TEST_F(TransformationTestsF, SliceToStridedSlice_axes_param) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 4, 3, 5});
        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        auto axes = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{4});

        auto slice = std::make_shared<ngraph::opset8::Slice>(data, begin, end, step, axes);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, axes});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    // No transformation for non-const axes input
}

TEST_F(TransformationTestsF, SliceToStridedSlice_begin_param_shape_of_use_shapes_true) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(data, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<ngraph::opset8::Range>(zero_const, one_const, one_const, ngraph::element::i64);

        auto slice = std::make_shared<ngraph::opset8::Slice>(shape_of_data, begin, end, step, axes);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin});

        manager.register_pass<ngraph::pass::StridedSliceOptimization>(true);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto data = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 3, 4, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto stride = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{begin});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_begin_param_shape_of_use_shapes_false) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(data, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<ngraph::opset8::Range>(zero_const, one_const, one_const, ngraph::element::i64);

        auto slice = std::make_shared<ngraph::opset8::Slice>(shape_of_data, begin, end, step, axes);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data, begin});

        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto data = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 3, 4, 5});
        auto begin = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto stride = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{begin});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_const_fold_params_slice_shape_of_use_shapes_true) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(data, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<ngraph::opset8::Range>(zero_const, one_const, one_const, ngraph::element::i64);

        auto slice = std::make_shared<ngraph::opset8::Slice>(shape_of_data, begin, end, step, axes);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(true);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto sliced_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 4});
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sliced_const}, ngraph::ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_const_fold_params_slice_shape_of_use_shapes_false) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(data, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto three_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});

        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto end = std::make_shared<ngraph::opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<ngraph::opset8::Range>(zero_const, one_const, one_const, ngraph::element::i64);

        auto slice = std::make_shared<ngraph::opset8::Slice>(shape_of_data, begin, end, step, axes);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto sliced_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {3, 4});
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sliced_const}, ngraph::ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_slice_all_use_shapes_true) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto relu = std::make_shared<ngraph::opset8::Relu>(data);

        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(relu, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});

        auto begin = std::make_shared<ngraph::opset8::Broadcast>(zero_const, data_rank);
        auto end = std::make_shared<ngraph::opset8::Broadcast>(data_rank, data_rank);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, data_rank);

        auto slice = std::make_shared<ngraph::opset8::Slice>(relu, begin, end, step);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(true);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto relu = std::make_shared<ngraph::opset8::Relu>(data);

        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {4});
        auto stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_end_mask = {0, 0, 0, 0};
        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(relu, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_slice_all_use_shapes_false) {
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto relu = std::make_shared<ngraph::opset8::Relu>(data);

        auto shape_of_data = std::make_shared<ngraph::opset8::ShapeOf>(relu, ngraph::element::i64);
        auto data_rank = std::make_shared<ngraph::opset8::ShapeOf>(shape_of_data, ngraph::element::i64);

        auto zero_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto one_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});

        auto begin = std::make_shared<ngraph::opset8::Broadcast>(zero_const, data_rank);
        auto end = std::make_shared<ngraph::opset8::Broadcast>(data_rank, data_rank);
        auto step = std::make_shared<ngraph::opset8::Broadcast>(one_const, data_rank);

        auto slice = std::make_shared<ngraph::opset8::Slice>(relu, begin, end, step);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{slice}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridedSliceOptimization>(false);
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
        auto relu = std::make_shared<ngraph::opset8::Relu>(data);

        auto begin = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0});
        auto end = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {4});
        auto stride = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_end_mask = {0, 0, 0, 0};
        auto strided_slice = std::make_shared<ngraph::opset8::StridedSlice>(relu, begin, end, stride, begin_end_mask, begin_end_mask);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{strided_slice}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
