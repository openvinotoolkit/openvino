// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gguf_dequantize.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::internal::GGUFDequantize;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, gguf_dequantize) {
    NodeBuilder::opset().insert<GGUFDequantize>();

    for (const auto format :
         {GGUFDequantize::Format::Q4_K, GGUFDequantize::Format::Q5_K, GGUFDequantize::Format::Q6_K}) {
        const auto bytes = static_cast<int64_t>(2 * GGUFDequantize::block_bytes(format));
        const auto data = std::make_shared<Parameter>(ov::element::u8, PartialShape{16, bytes});
        const auto op = std::make_shared<GGUFDequantize>(data, format, ov::element::f16);

        NodeBuilder builder(op, {data});
        auto g_op = ov::as_type_ptr<GGUFDequantize>(builder.create());

        EXPECT_EQ(g_op->get_format(), op->get_format());
        EXPECT_EQ(g_op->get_destination_type(), op->get_destination_type());
        EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
        EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    }
}
