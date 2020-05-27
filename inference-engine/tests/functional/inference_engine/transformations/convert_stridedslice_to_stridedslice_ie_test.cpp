// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_strided_slice_to_strided_slice_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/strided_slice_ie.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertStridedSliceToStridedSliceIEStatic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask{0, 0, 0, 0};
        std::vector<int64_t> end_mask{1, 1, 1, 1};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertStridedSliceToStridedSliceIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask{0, 0, 0, 0}, end_mask{1, 1, 1, 1}, new_axis_mask{}, shrink_axis_mask{}, ellipsis_mask{};

        auto ss = std::make_shared<ngraph::op::StridedSliceIE>(data, begin, end, stride,
                begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertStridedSliceToStridedSliceIEDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask{0, 0, 0, 0};
        std::vector<int64_t> end_mask{1, 1, 1, 1};

        auto ss = std::make_shared<ngraph::opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertStridedSliceToStridedSliceIE().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto begin = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {0, 0, 0, 0});
        auto end = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {-1, -1, -1, -1});
        auto stride = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1});

        std::vector<int64_t> begin_mask{0, 0, 0, 0}, end_mask{1, 1, 1, 1}, new_axis_mask{}, shrink_axis_mask{}, ellipsis_mask{};

        auto ss = std::make_shared<ngraph::op::StridedSliceIE>(data, begin, end, stride,
                begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
