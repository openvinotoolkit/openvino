// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, paged_attention) {
    NodeBuilder::opset().insert<ov::op::internal::PagedAttentionExtension>();
    const auto data1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data5 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data6 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data7 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data8 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data9 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data10 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data11 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data12 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    const auto data13 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});

    const auto paged_attention = std::make_shared<ov::op::internal::PagedAttentionExtension>(data1,
                                                                                             data2,
                                                                                             data3,
                                                                                             data4,
                                                                                             data5,
                                                                                             data6,
                                                                                             data7,
                                                                                             data8,
                                                                                             data9,
                                                                                             data10,
                                                                                             data11,
                                                                                             data12,
                                                                                             data13);
    NodeBuilder builder(
        paged_attention,
        {data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13});
    auto g_paged_attention = ov::as_type_ptr<ov::op::internal::PagedAttentionExtension>(builder.create());

    constexpr auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_paged_attention->get_out_type(0), paged_attention->get_out_type(0));
    EXPECT_EQ(g_paged_attention->get_out_type(1), paged_attention->get_out_type(1));
}
