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

TEST(attributes, convolution) {
    NodeBuilder::get_ops().register_factory<op::v1::Convolution>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 124, 124});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 16, 3, 3});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto convolution =
        make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, op::PadType::VALID);

    NodeBuilder builder(convolution);
    auto g_convolution = ov::as_type_ptr<op::v1::Convolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
}
