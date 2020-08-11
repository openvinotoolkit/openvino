// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <transformations/convert_batch_to_space.hpp>
#include <transformations/convert_space_to_batch.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, BatchToSpaceDecompositionByElements) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertBatchToSpace>();
        m.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {10, 10, 7, 13, 3});
        auto axis_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 0, 3, 4});
        auto squeezed_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {10, 70, 13, 3});

        auto reshape_before_1 = std::make_shared<ngraph::opset3::Reshape> (data, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<ngraph::opset3::Reshape> (permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {5, 2, 70, 13, 3});
        auto axis_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 3, 0, 4});
        auto squeezed_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_2 = std::make_shared<ngraph::opset3::Reshape> (reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<ngraph::opset3::Reshape> (permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 70, 65, 3});
        auto axis_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 2, 3, 4, 0});
        auto squeezed_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_3 = std::make_shared<ngraph::opset3::Reshape> (reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<ngraph::opset3::Reshape> (permute_3, squeezed_order_3, false);

        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after_3, begin, end, begin_mask, end_mask);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToBatchDecompositionByElements) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertSpaceToBatch>();
        m.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, ngraph::op::PadMode::CONSTANT);

        auto dispresed_shape_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 70, 65, 3, 1});
        auto axis_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {4, 0, 1, 2, 3});
        auto squeezed_order_1 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 70, 65, 3});

        auto reshape_before_1 = std::make_shared<ngraph::opset3::Reshape> (pads, dispresed_shape_1, false);
        auto permute_1 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_1, axis_order_1);
        auto reshape_after_1 = std::make_shared<ngraph::opset3::Reshape> (permute_1, squeezed_order_1, false);

        auto dispresed_shape_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 70, 13, 5, 3});
        auto axis_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {3, 0, 1, 2, 4});
        auto squeezed_order_2 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {10, 70, 13, 3});

        auto reshape_before_2 = std::make_shared<ngraph::opset3::Reshape> (reshape_after_1, dispresed_shape_2, false);
        auto permute_2 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_2, axis_order_2);
        auto reshape_after_2 = std::make_shared<ngraph::opset3::Reshape> (permute_2, squeezed_order_2, false);

        auto dispresed_shape_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {10, 7, 10, 13, 3});
        auto axis_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {2, 0, 1, 3, 4});
        auto squeezed_order_3 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {100, 7, 13, 3});

        auto reshape_before_3 = std::make_shared<ngraph::opset3::Reshape> (reshape_after_2, dispresed_shape_3, false);
        auto permute_3 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_3, axis_order_3);
        auto reshape_after_3 = std::make_shared<ngraph::opset3::Reshape> (permute_3, squeezed_order_3, false);

        auto dispresed_shape_4 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {100, 1, 7, 13, 3});
        auto axis_order_4 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{5}, {1, 0, 2, 3, 4});
        auto squeezed_order_4 = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {100, 7, 13, 3});

        auto reshape_before_4 = std::make_shared<ngraph::opset3::Reshape> (reshape_after_3, dispresed_shape_4, false);
        auto permute_4 = std::make_shared<ngraph::opset3::Transpose> (reshape_before_4, axis_order_4);
        auto reshape_after_4 = std::make_shared<ngraph::opset3::Reshape> (permute_4, squeezed_order_4, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after_4}, ngraph::ParameterVector{data});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToBatchDecomposition) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto block_shape = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 10, 5, 1});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertSpaceToBatch>(false);
        m.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{2, 64, 64, 3});
        auto pads_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto pads_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto pads = std::make_shared<opset3::Pad>(data, pads_begin, pads_end, ngraph::op::PadMode::CONSTANT);

        auto dispresed_shape = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7},
                                                                  {2, 7, 10, 13, 5, 3, 1});
        auto axis_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {2, 4, 6, 0, 1, 3, 5});
        auto squeezed_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4},
                                                                 {100, 7, 13, 3});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(pads, dispresed_shape, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, squeezed_order, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{data});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BatchToSpaceDecomposition) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});
        auto block_shape = std::make_shared<opset3::Constant>(element::i64, Shape{4},
                                                              std::vector<int64_t>{1, 10, 5, 1});
        auto crops_begin = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 1, 0});
        auto crops_end = std::make_shared<opset3::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 3, 0, 0});
        auto batch_to_space = std::make_shared<ngraph::opset3::BatchToSpace>(data, block_shape, crops_begin, crops_end);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_to_space}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertBatchToSpace>(false);
        m.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{100, 7, 13, 3});

        auto dispresed_shape = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7},
                                                                  {10, 5, 1, 2, 7, 13, 3});
        auto axis_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{7}, {3, 4, 0, 5, 1, 6, 2});
        auto squeezed_order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4},
                                                                 {2, 70, 65, 3});

        auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(data, dispresed_shape, false);
        auto permute = std::make_shared<ngraph::opset3::Transpose>(reshape_before, axis_order);
        auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(permute, squeezed_order, false);

        auto begin = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 0});
        auto end = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 67, 65, 3});
        std::vector<int64_t> begin_mask(4, 0);
        std::vector<int64_t> end_mask(4, 0);
        auto ss = std::make_shared<opset3::StridedSlice>(reshape_after, begin, end, begin_mask, end_mask);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{data});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
