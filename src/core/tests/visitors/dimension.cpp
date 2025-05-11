// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/attribute_visitor.hpp"
#include "visitors/visitors.hpp"

using namespace std;

TEST(attributes, dimension) {
    ov::test::NodeBuilder builder;
    ov::AttributeVisitor& loader = builder.get_node_loader();
    ov::AttributeVisitor& saver = builder.get_node_saver();

    ov::Dimension dyn = ov::Dimension(-1);
    saver.on_attribute("dyn", dyn);
    ov::Dimension g_dyn;
    loader.on_attribute("dyn", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    ov::Dimension scalar = ov::Dimension(10);
    saver.on_attribute("scalar", scalar);
    ov::Dimension g_scalar;
    loader.on_attribute("scalar", g_scalar);
    EXPECT_EQ(scalar, g_scalar);

    ov::Dimension boundaries1 = ov::Dimension(2, 100);
    saver.on_attribute("boundaries1", boundaries1);
    ov::Dimension g_boundaries1;
    loader.on_attribute("boundaries1", g_boundaries1);
    EXPECT_EQ(boundaries1, g_boundaries1);

    ov::Dimension boundaries2 = ov::Dimension(-1, 100);
    saver.on_attribute("boundaries2", boundaries2);
    ov::Dimension g_boundaries2;
    loader.on_attribute("boundaries2", g_boundaries2);
    EXPECT_EQ(boundaries2, g_boundaries2);

    ov::Dimension boundaries3 = ov::Dimension(5, -1);
    saver.on_attribute("boundaries3", boundaries3);
    ov::Dimension g_boundaries3;
    loader.on_attribute("boundaries3", g_boundaries3);
    EXPECT_EQ(boundaries3, g_boundaries3);
}
