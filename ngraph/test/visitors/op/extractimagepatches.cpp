// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/extractimagepatches.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, extractimagepatches_op) {
    NodeBuilder::get_ops().register_factory<op::v3::ExtractImagePatches>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});

    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = ngraph::op::PadType::VALID;

    auto extractimagepatches = make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);
    NodeBuilder builder(extractimagepatches);
    auto g_extractimagepatches = as_type_ptr<op::v3::ExtractImagePatches>(builder.create());

    const auto expected_attr_count = 4;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_extractimagepatches->get_sizes(), sizes);
    EXPECT_EQ(g_extractimagepatches->get_strides(), strides);
    EXPECT_EQ(g_extractimagepatches->get_rates(), rates);
    EXPECT_EQ(g_extractimagepatches->get_auto_pad(), padtype_padding);
}
