// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <legacy/ngraph_ops/tile_ie.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertTileToLegacyDynamic1) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, Dimension::dynamic()});
    auto axes = opset1::Constant::create(element::i64, Shape{1}, {0});
    auto tile = std::make_shared<opset1::Tile>(data, axes);

    auto f = std::make_shared<Function>(NodeVector{tile}, ParameterVector{data});
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertTileToLegacyMatcher>();
    ASSERT_NO_THROW(manager.run_passes(f));
    ASSERT_NO_THROW(check_rt_info(f));
}

TEST(TransformationTests, ConvertTileToLegacyDynamic2) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = opset1::Constant::create(element::i64, Shape{1}, {0});
    auto tile = std::make_shared<opset1::Tile>(data, axes);

    auto f = std::make_shared<Function>(NodeVector{tile}, ParameterVector{data});
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertTileToLegacyMatcher>();
    ASSERT_NO_THROW(manager.run_passes(f));
    ASSERT_NO_THROW(check_rt_info(f));
}