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

TEST(attributes, reverse_sequence_op) {
    NodeBuilder::get_ops().register_factory<opset1::ReverseSequence>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 2});
    auto seq_indices = make_shared<op::Parameter>(element::i32, Shape{4});

    auto batch_axis = 2;
    auto seq_axis = 1;

    auto reverse_sequence = make_shared<opset1::ReverseSequence>(data, seq_indices, batch_axis, seq_axis);

    NodeBuilder builder(reverse_sequence);
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto g_reverse_sequence = ov::as_type_ptr<opset1::ReverseSequence>(builder.create());

    EXPECT_EQ(g_reverse_sequence->get_origin_batch_axis(), reverse_sequence->get_origin_batch_axis());
    EXPECT_EQ(g_reverse_sequence->get_origin_sequence_axis(), reverse_sequence->get_origin_sequence_axis());
}
