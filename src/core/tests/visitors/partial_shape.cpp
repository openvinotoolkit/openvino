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

TEST(attributes, partial_shape) {
    NodeBuilder builder;
    AttributeVisitor& loader = builder.get_node_loader();
    AttributeVisitor& saver = builder.get_node_saver();

    PartialShape dyn = PartialShape::dynamic();
    saver.on_attribute("dyn", dyn);
    PartialShape g_dyn;
    loader.on_attribute("dyn", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    PartialShape scalar{};
    saver.on_attribute("scalar", scalar);
    PartialShape g_scalar;
    loader.on_attribute("scalar", g_scalar);
    EXPECT_EQ(scalar, g_scalar);

    PartialShape dyn_vector{Dimension::dynamic()};
    saver.on_attribute("dyn_vector", dyn_vector);
    PartialShape g_dyn_vector;
    loader.on_attribute("dyn_vector", g_dyn_vector);
    EXPECT_EQ(dyn_vector, g_dyn_vector);

    PartialShape stat_vector{7};
    saver.on_attribute("stat_vector", stat_vector);
    PartialShape g_stat_vector;
    loader.on_attribute("stat_vector", g_stat_vector);
    EXPECT_EQ(stat_vector, g_stat_vector);

    PartialShape general{7, Dimension::dynamic(), 2, Dimension::dynamic(), 4};
    saver.on_attribute("general", general);
    PartialShape g_general;
    loader.on_attribute("general", g_general);
    EXPECT_EQ(general, g_general);

    PartialShape shape_with_boundaries{Dimension(2, 20)};
    saver.on_attribute("shape_with_boundaries", shape_with_boundaries);
    PartialShape g_shape_with_boundaries;
    loader.on_attribute("shape_with_boundaries", g_shape_with_boundaries);
    EXPECT_EQ(shape_with_boundaries, g_shape_with_boundaries);

    PartialShape shape_with_undefined_boundaries{Dimension(10, -1), Dimension(-1, 100), Dimension::dynamic(), 4};
    saver.on_attribute("shape_with_undefined_boundaries", shape_with_undefined_boundaries);
    PartialShape g_shape_with_undefined_boundaries;
    loader.on_attribute("shape_with_undefined_boundaries", g_shape_with_undefined_boundaries);
    EXPECT_EQ(shape_with_undefined_boundaries, g_shape_with_undefined_boundaries);
}
