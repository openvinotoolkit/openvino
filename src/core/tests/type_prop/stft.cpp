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

using STFTTestParam = std::tuple<PartialShape, PartialShape, int32_t, int32_t, bool, PartialShape>;
class TypePropSTFTTestP : public TypePropSTFTTest, public testing::WithParamInterface<STFTTestParam> {
protected:
    void SetUp() override {
        std::tie(signal_shape, window_shape, frame_size_val, step_size_val, transform_frames, expected_shape) =
            GetParam();
    }
    PartialShape signal_shape, window_shape, expected_shape;
    int32_t frame_size_val, step_size_val;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_stft_shape,
    TypePropSTFTTestP,
    testing::Values(
        std::make_tuple(PartialShape{4, 48}, PartialShape{7}, 11, 3, true, PartialShape{4, 6, 13, 2}),
        std::make_tuple(PartialShape{4, 48}, PartialShape{7}, 11, 3, false, PartialShape{4, 13, 6, 2}),
        std::make_tuple(PartialShape{4, 56}, PartialShape{7}, 11, 3, true, PartialShape{4, 6, 16, 2}),
        std::make_tuple(PartialShape{4, 56}, PartialShape{7}, 11, 3, false, PartialShape{4, 16, 6, 2}),
        std::make_tuple(PartialShape{-1, 56}, PartialShape{7}, 11, 3, false, PartialShape{-1, 16, 6, 2}),
        std::make_tuple(PartialShape{-1, -1}, PartialShape{7}, 11, 3, false, PartialShape{-1, {1, -1}, 6, 2}),
        std::make_tuple(PartialShape{-1, -1}, PartialShape{7}, 11, 3, true, PartialShape{-1, 6, {1, -1}, 2}),
        std::make_tuple(PartialShape{-1, {48, 56}}, PartialShape{7}, 11, 3, true, PartialShape{-1, 6, {13, 16}, 2}),
        std::make_tuple(PartialShape::dynamic(), PartialShape{7}, 11, 3, true, PartialShape::dynamic()),
        std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), 11, 3, true, PartialShape::dynamic())),
    testing::PrintToStringParamName());

TEST_P(TypePropSTFTTestP, stft_shapes) {
    const auto signal = std::make_shared<Parameter>(element::f32, signal_shape);
    const auto window = std::make_shared<Parameter>(ov::element::f32, window_shape);
    const auto frame_size = Constant::create<int32_t>(ov::element::i32, {}, {frame_size_val});
    const auto frame_step = Constant::create<int32_t>(ov::element::i32, {}, {step_size_val});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

}  // namespace test
}  // namespace ov
