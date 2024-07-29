// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropSTFTTest : public TypePropOpTest<op::v15::STFT> {
public:
    bool transform_frames = true;
};

TEST_F(TypePropSTFTTest, default_ctor) {
    const auto op = make_op();
    const auto signal = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 48});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{7});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {11});
    const auto frame_step = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {3});

    op->set_arguments(ov::OutputVector{signal, window, frame_size, frame_step});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 13, 6, 2}));
}

TEST_F(TypePropSTFTTest, fully_dynamic_data_shape) {
    const auto signal = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{7});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {11});
    const auto frame_step = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {3});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropSTFTTest, dynamic_data_shape_static_rank) {
    const auto signal = std::make_shared<Parameter>(element::f16, ov::PartialShape{-1, -1});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{7});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {11});
    const auto frame_step = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {3});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    // Static dimension is calculated for the second dim based on the frame_size val, (11/2) + 1 = 6
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{-1, 6, {1, -1}, 2}));
}

TEST_F(TypePropSTFTTest, dynamic_data_shape_range) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{-1, {48, 56}});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{7});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {11});
    const auto frame_step = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {3});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 6, {13, 16}, 2}));
}

TEST_F(TypePropSTFTTest, dynamic_data_shape_rank) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{7});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {11});
    const auto frame_step = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {3});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}
}  // namespace test
}  // namespace ov
