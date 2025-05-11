// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, fake_quantize) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const size_t levels = 5;

    const auto fake_quantize =
        make_shared<op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    EXPECT_EQ(fake_quantize->get_element_type(), element::f32);
    EXPECT_EQ(fake_quantize->get_shape(), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, fake_quantize_autob) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1});
    const auto input_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto output_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    const auto output_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const size_t levels = 5;

    const auto fake_quantize =
        make_shared<op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    EXPECT_EQ(fake_quantize->get_element_type(), element::f32);
    EXPECT_EQ(fake_quantize->get_shape(), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, fake_quantize_invalid_autob) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto input_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto input_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto output_low = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto output_high = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const size_t levels = 5;

    try {
        const auto fake_quantize =
            make_shared<op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
}
