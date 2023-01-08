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

TEST(attributes, value_map) {
    ValueMap value_map;
    bool a = true;
    int8_t b = 2;
    value_map.insert("a", a);
    value_map.insert("b", b);
    bool g_a = value_map.get<bool>("a");
    int8_t g_b = value_map.get<int8_t>("b");
    EXPECT_EQ(a, g_a);
    EXPECT_EQ(b, g_b);
}
