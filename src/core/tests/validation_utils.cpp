// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/type.hpp>
#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/util/common_util.hpp>

TEST(get_constant_from_source, invalidation_check) {
    auto a = ov::opset8::Constant::create(ov::element::i64, {100}, {123});
    auto b = ov::opset8::Constant::create(ov::element::i64, {1}, {123});
    auto div = std::make_shared<ov::opset8::Divide>(a, b);
    auto s = std::make_shared<ov::opset8::ShapeOf>(a);
    auto r = std::make_shared<ov::opset8::Reshape>(div, s, true);
    auto tmp_consumer = std::make_shared<ov::opset8::ShapeOf>(s);

    ASSERT_TRUE(ov::get_constant_from_source(r));

    ASSERT_TRUE(r->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(r->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(s->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(s->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(b->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(b->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(a->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(a->get_output_tensor(0).get_upper_value());

    ASSERT_FALSE(div->get_output_tensor(0).get_lower_value());
    ASSERT_FALSE(div->get_output_tensor(0).get_upper_value());
}
