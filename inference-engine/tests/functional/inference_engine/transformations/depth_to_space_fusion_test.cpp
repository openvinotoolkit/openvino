// Copyright (C) 2020 Intel Corporation
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
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <transformations/depth_to_space_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, DepthToSpaceFusionDepthFirst) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 32, 2, 2, 720, 480});
        auto permutation = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 1, 4, 2, 5, 3});
        auto shape_reshape_after = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset1::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset1::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset1::Reshape> (permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return true;
        };

        auto depth_to_space_transform = ngraph::pass::DepthToSpaceFusion();
        depth_to_space_transform.setCallback(callback);
        depth_to_space_transform.run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space = std::make_shared<ngraph::opset1::DepthToSpace>(input0, ngraph::opset1::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DepthToSpaceFusionBlockFirst) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto shape_reshape_before = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {1, 2, 2, 32, 720, 480});
        auto permutation = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{6}, {0, 3, 4, 1, 5, 2});
        auto shape_reshape_after = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 32, 1440, 960});

        auto reshape_before = std::make_shared<ngraph::opset1::Reshape> (input0, shape_reshape_before, false);
        auto permute = std::make_shared<ngraph::opset1::Transpose> (reshape_before, permutation);
        auto reshape_after = std::make_shared<ngraph::opset1::Reshape> (permute, shape_reshape_after, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_after}, ngraph::ParameterVector{input0});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            return true;
        };

        auto depth_to_space_transform = ngraph::pass::DepthToSpaceFusion();
        depth_to_space_transform.setCallback(callback);
        depth_to_space_transform.run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 720, 480});
        auto depth_to_space = std::make_shared<ngraph::opset1::DepthToSpace>(input0, ngraph::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
