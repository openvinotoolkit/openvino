// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertBroadcastToTilesDynamic) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape);
        broadcast->set_friendly_name("broadcast");

        auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertBroadcastToTiles>();
        ASSERT_NO_THROW(manager.run_passes(f));
        ASSERT_NO_THROW(check_rt_info(f));
    }
    // TODO: construct reference graph and use TEST_F
}

