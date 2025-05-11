// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/attribute_visitor.hpp"
#include "visitors/visitors.hpp"

using namespace std;

TEST(attributes, partial_shape) {
    ov::test::NodeBuilder builder;
    ov::AttributeVisitor& loader = builder.get_node_loader();
    ov::AttributeVisitor& saver = builder.get_node_saver();

    ov::PartialShape dyn = ov::PartialShape::dynamic();
    saver.on_attribute("dyn", dyn);
    ov::PartialShape g_dyn;
    loader.on_attribute("dyn", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    ov::PartialShape scalar{};
    saver.on_attribute("scalar", scalar);
    ov::PartialShape g_scalar;
    loader.on_attribute("scalar", g_scalar);
    EXPECT_EQ(scalar, g_scalar);

    ov::PartialShape dyn_vector{ov::Dimension::dynamic()};
    saver.on_attribute("dyn_vector", dyn_vector);
    ov::PartialShape g_dyn_vector;
    loader.on_attribute("dyn_vector", g_dyn_vector);
    EXPECT_EQ(dyn_vector, g_dyn_vector);

    ov::PartialShape stat_vector{7};
    saver.on_attribute("stat_vector", stat_vector);
    ov::PartialShape g_stat_vector;
    loader.on_attribute("stat_vector", g_stat_vector);
    EXPECT_EQ(stat_vector, g_stat_vector);

    ov::PartialShape general{7, ov::Dimension::dynamic(), 2, ov::Dimension::dynamic(), 4};
    saver.on_attribute("general", general);
    ov::PartialShape g_general;
    loader.on_attribute("general", g_general);
    EXPECT_EQ(general, g_general);

    ov::PartialShape shape_with_boundaries{ov::Dimension(2, 20)};
    saver.on_attribute("shape_with_boundaries", shape_with_boundaries);
    ov::PartialShape g_shape_with_boundaries;
    loader.on_attribute("shape_with_boundaries", g_shape_with_boundaries);
    EXPECT_EQ(shape_with_boundaries, g_shape_with_boundaries);

    ov::PartialShape shape_with_undefined_boundaries{ov::Dimension(10, -1),
                                                     ov::Dimension(-1, 100),
                                                     ov::Dimension::dynamic(),
                                                     4};
    saver.on_attribute("shape_with_undefined_boundaries", shape_with_undefined_boundaries);
    ov::PartialShape g_shape_with_undefined_boundaries;
    loader.on_attribute("shape_with_undefined_boundaries", g_shape_with_undefined_boundaries);
    EXPECT_EQ(shape_with_undefined_boundaries, g_shape_with_undefined_boundaries);
}
