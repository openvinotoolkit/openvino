// Copyright (C) 2018-2025 Intel Corporation
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

using op::v0::Constant;
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropSTFTTest : public TypePropOpTest<op::v15::STFT> {
public:
    bool transform_frames = true;
};

TEST_F(TypePropSTFTTest, default_ctor) {
    const auto op = make_op();
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {11});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});

    op->set_arguments(OutputVector{signal, window, frame_size, frame_step});
    op->set_transpose_frames(true);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 6, 13, 2}));
}

TEST_F(TypePropSTFTTest, all_inputs_as_params_static_shapes) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, -1, -1, 2}));
}

TEST_F(TypePropSTFTTest, all_inputs_as_params_f16_i32_static_shapes) {
    const auto signal = std::make_shared<Parameter>(element::f16, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f16, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i32, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i32, PartialShape{});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, -1, -1, 2}));
}

TEST_F(TypePropSTFTTest, all_inputs_as_params_dyn_shapes) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{-1});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1, -1, 2}));
}

TEST_F(TypePropSTFTTest, all_inputs_as_params_dyn_rank) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
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
        std::make_tuple(PartialShape{16}, PartialShape{16}, 16, 16, true, PartialShape{9, 1, 2}),
        std::make_tuple(PartialShape{48}, PartialShape{16}, 16, 16, false, PartialShape{3, 9, 2}),
        std::make_tuple(PartialShape{56}, PartialShape{7}, 11, 3, false, PartialShape{16, 6, 2}),
        std::make_tuple(PartialShape{56}, PartialShape{7}, 11, 3, true, PartialShape{6, 16, 2}),
        std::make_tuple(PartialShape{48}, PartialShape{8}, 16, 4, true, PartialShape{9, 9, 2}),
        std::make_tuple(PartialShape{{48, 56}}, PartialShape{7}, 11, 3, true, PartialShape{6, {13, 16}, 2}),
        std::make_tuple(PartialShape{-1}, PartialShape{7}, 11, 3, true, PartialShape{6, {1, -1}, 2}),
        std::make_tuple(PartialShape{1, 16}, PartialShape{16}, 16, 16, true, PartialShape{1, 9, 1, 2}),
        std::make_tuple(PartialShape{1, 48}, PartialShape{16}, 16, 16, true, PartialShape{1, 9, 3, 2}),
        std::make_tuple(PartialShape{1, 48}, PartialShape{16}, 16, 16, false, PartialShape{1, 3, 9, 2}),
        std::make_tuple(PartialShape{2, 48}, PartialShape{8}, 16, 4, true, PartialShape{2, 9, 9, 2}),
        std::make_tuple(PartialShape{2, 48}, PartialShape{5}, 9, 100, true, PartialShape{2, 5, 1, 2}),
        std::make_tuple(PartialShape{4, 48}, PartialShape{7}, 11, 3, true, PartialShape{4, 6, 13, 2}),
        std::make_tuple(PartialShape{4, 48}, PartialShape{7}, 11, 3, false, PartialShape{4, 13, 6, 2}),
        std::make_tuple(PartialShape{4, 56}, PartialShape{7}, 11, 3, true, PartialShape{4, 6, 16, 2}),
        std::make_tuple(PartialShape{4, 56}, PartialShape{7}, 11, 3, false, PartialShape{4, 16, 6, 2}),
        std::make_tuple(PartialShape{-1, 56}, PartialShape{7}, 11, 3, false, PartialShape{-1, 16, 6, 2}),
        std::make_tuple(PartialShape{-1, -1}, PartialShape{7}, 11, 3, false, PartialShape{-1, {1, -1}, 6, 2}),
        std::make_tuple(PartialShape{-1, -1}, PartialShape{7}, 11, 3, true, PartialShape{-1, 6, {1, -1}, 2}),
        std::make_tuple(PartialShape{-1, {48, 56}}, PartialShape{7}, 11, 3, true, PartialShape{-1, 6, {13, 16}, 2}),
        std::make_tuple(PartialShape{{2, 4}, -1}, PartialShape{7}, 11, 3, true, PartialShape{{2, 4}, 6, {1, -1}, 2}),
        std::make_tuple(PartialShape::dynamic(), PartialShape{7}, 11, 3, true, PartialShape::dynamic()),
        std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), 11, 3, true, PartialShape::dynamic())),
    testing::PrintToStringParamName());

TEST_P(TypePropSTFTTestP, stft_shapes) {
    const auto signal = std::make_shared<Parameter>(element::f32, signal_shape);
    const auto window = std::make_shared<Parameter>(element::f32, window_shape);
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {frame_size_val});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {step_size_val});

    const auto op = make_op(signal, window, frame_size, frame_step, transform_frames);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TEST_F(TypePropSTFTTest, signal_incompatible_shape) {
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of signal must be 1D [signal_size] or 2D [batch, signal_size]"));
    }
    {
        const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{-1, 4, 48});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of signal must be 1D [signal_size] or 2D [batch, signal_size]"));
    }
}

TEST_F(TypePropSTFTTest, window_incompatible_shape) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto window = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of window must be 1D [window_size]"));
    }
    {
        const auto window = std::make_shared<Parameter>(element::f32, PartialShape{2, 8});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of window must be 1D [window_size]"));
    }
}

TEST_F(TypePropSTFTTest, frame_size_incompatible_shape) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_size must be a scalar"));
    }
    {
        const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_size must be a scalar"));
    }
}

TEST_F(TypePropSTFTTest, frame_step_incompatible_shape) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{1});

        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_step must be a scalar"));
    }
    {
        const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_step must be a scalar"));
    }
}

TEST_F(TypePropSTFTTest, signal_incompatible_type) {
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto signal = std::make_shared<Parameter>(element::i32, PartialShape{1, 48});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected floating point type of the 'signal' input"));
    }
    {
        const auto signal = std::make_shared<Parameter>(element::boolean, PartialShape{1, 48});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected floating point type of the 'signal' input"));
    }
}

TEST_F(TypePropSTFTTest, window_incompatible_type) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto window = std::make_shared<Parameter>(element::i32, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `signal` input"));
    }
    {
        const auto window = std::make_shared<Parameter>(element::boolean, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `signal` input"));
    }
    {
        // Doesn't match the type of signal input
        const auto window = std::make_shared<Parameter>(element::f16, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `signal` input"));
    }
}

TEST_F(TypePropSTFTTest, frame_size_incompatible_type) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_size = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 2"));
    }
    {
        const auto frame_size = std::make_shared<Parameter>(element::i8, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 2"));
    }
}

TEST_F(TypePropSTFTTest, frame_step_incompatible_type) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_step = std::make_shared<Parameter>(element::f32, PartialShape{});

        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 3"));
    }
    {
        const auto frame_step = std::make_shared<Parameter>(element::i8, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 3"));
    }
}

TEST_F(TypePropSTFTTest, frame_size_incompatible_value) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});
    {
        const auto frame_size = Constant::create<int32_t>(element::i32, {}, {-1});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Provided frame size is -1 but must be in range [1, 48]"));
    }
    {
        const auto frame_size = Constant::create<int32_t>(element::i32, {}, {49});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Provided frame size is 49 but must be in range [1, 48]"));
    }
}

TEST_F(TypePropSTFTTest, frame_step_incompatible_value) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {8});
    {
        const auto frame_step = Constant::create<int32_t>(element::i32, {}, {-1});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                        NodeValidationFailure,
                        HasSubstr("Provided frame step is -1 but must be greater than zero"));
    }
}

TEST_F(TypePropSTFTTest, window_incompatible_dim_with_frame_size) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {8});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {4});
    OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, transform_frames),
                    NodeValidationFailure,
                    HasSubstr("Window input dimension must be in range [1, 8]"));
}
}  // namespace test
}  // namespace ov
