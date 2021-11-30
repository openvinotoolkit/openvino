// Copyright (C) 2018-2021 Intel Corporation
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

TEST(attributes, dimension) {
    NodeBuilder builder;
    AttributeVisitor& loader = builder.get_node_loader();
    AttributeVisitor& saver = builder.get_node_saver();

    Dimension dyn = Dimension(-1);
    saver.on_attribute("dyn", dyn);
    Dimension g_dyn;
    loader.on_attribute("dyn", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    Dimension scalar = Dimension(10);
    saver.on_attribute("scalar", dyn);
    Dimension g_scalar;
    loader.on_attribute("scalar", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    Dimension boundaries1 = Dimension(2, 100);
    saver.on_attribute("boundaries1", dyn);
    Dimension g_boundaries1;
    loader.on_attribute("boundaries1", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    Dimension boundaries2 = Dimension(-1, 100);
    saver.on_attribute("boundaries2", dyn);
    Dimension g_boundaries2;
    loader.on_attribute("boundaries2", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    Dimension boundaries3 = Dimension(5, -1);
    saver.on_attribute("boundaries3", dyn);
    Dimension g_boundaries3;
    loader.on_attribute("boundaries3", g_dyn);
    EXPECT_EQ(dyn, g_dyn);
}
