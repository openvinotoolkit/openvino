// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, space_to_depth_op) {
    NodeBuilder::get_ops().register_factory<opset1::SpaceToDepth>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 50, 50});

    auto block_size = 2;
    auto mode = opset1::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;

    auto space_to_depth = make_shared<opset1::SpaceToDepth>(data, mode, block_size);
    NodeBuilder builder(space_to_depth);
    auto g_space_to_depth = ov::as_type_ptr<opset1::SpaceToDepth>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // space_to_depth attributes
    EXPECT_EQ(g_space_to_depth->get_block_size(), space_to_depth->get_block_size());
    EXPECT_EQ(g_space_to_depth->get_mode(), space_to_depth->get_mode());
}
