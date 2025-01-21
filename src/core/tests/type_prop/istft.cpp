// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

using op::v0::Constant;
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropISTFTTest : public TypePropOpTest<op::v16::ISTFT> {
public:
    bool center = true;
    bool normalized = false;
};

TEST_F(TypePropISTFTTest, all_inputs_as_params_static_shapes) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{2, 1, 4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto length = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(in_data, window, frame_size, frame_step, length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, -1}));
}

using STFTTestParam = std::tuple<PartialShape, PartialShape, int32_t, int32_t, bool, PartialShape>;
class TypePropISTFTTestP : public TypePropISTFTTest, public testing::WithParamInterface<STFTTestParam> {
protected:
    void SetUp() override {
        std::tie(signal_shape, window_shape, frame_size_val, step_size_val, center, expected_shape) = GetParam();
    }
    PartialShape signal_shape, window_shape, expected_shape;
    int32_t frame_size_val, step_size_val;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_stft_shape,
    TypePropISTFTTestP,
    testing::Values(
        std::make_tuple(PartialShape{16}, PartialShape{16}, 16, 16, false, PartialShape{9, 1, 2}), // frames at 1
        std::make_tuple(PartialShape{48}, PartialShape{16}, 16, 16, false, PartialShape{9, 3, 2}),
        std::make_tuple(PartialShape{56}, PartialShape{7}, 11, 3, false, PartialShape{6, 16, 2}),

        std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), 11, 3, true, PartialShape::dynamic())),
    testing::PrintToStringParamName());

TEST_P(TypePropISTFTTestP, istft_shapes) {
    const auto in_data = std::make_shared<Parameter>(element::f32, expected_shape);
    const auto window = std::make_shared<Parameter>(element::f32, window_shape);
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {frame_size_val});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {step_size_val});
    const auto length = std::make_shared<Parameter>(element::i32, Shape{});

    const auto op = make_op(in_data, window, frame_size, frame_step, length, center, false);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), signal_shape);
}

}  // namespace test
}  // namespace ov
