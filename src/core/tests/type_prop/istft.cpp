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

namespace ov::test {

using op::v0::Constant;
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropISTFTTest : public TypePropOpTest<op::v16::ISTFT> {
public:
    bool center = false;
    bool normalized = false;
};

TEST_F(TypePropISTFTTest, default_ctor) {
    const auto op = make_op();
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 6, 16, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {11});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});

    op->set_arguments(OutputVector{data, window, frame_size, frame_step});
    op->validate_and_infer_types();

    op->set_center(true);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 56}));
}

TEST_F(TypePropISTFTTest, all_inputs_as_params_static_shapes_auto_length) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{2, 1, 4, 48});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(in_data, window, frame_size, frame_step, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, -1}));
}

TEST_F(TypePropISTFTTest, all_inputs_as_params_static_shapes_in_length) {
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

TEST_F(TypePropISTFTTest, all_inputs_as_params_f16_static_shapes) {
    const auto in_data = std::make_shared<Parameter>(element::f16, PartialShape{2, 1, 4, 48});
    const auto window = std::make_shared<Parameter>(element::f16, PartialShape{7});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(in_data, window, frame_size, frame_step, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, -1}));
}

TEST_F(TypePropISTFTTest, all_inputs_as_params_dyn_shapes) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{-1});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto length = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(in_data, window, frame_size, frame_step, length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1}));
}

TEST_F(TypePropISTFTTest, all_inputs_as_params_dyn_rank) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto length = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(in_data, window, frame_size, frame_step, length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

using STFTTestParam = std::tuple<PartialShape, PartialShape, int32_t, int32_t, bool, PartialShape>;
class TypePropISTFTTestP : public TypePropISTFTTest, public testing::WithParamInterface<STFTTestParam> {
protected:
    void SetUp() override {
        std::tie(signal_shape, window_shape, frame_size_val, step_size_val, center, data_shape) = GetParam();
    }
    PartialShape signal_shape, window_shape, data_shape;
    int32_t frame_size_val, step_size_val;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_stft_shape,
    TypePropISTFTTestP,
    testing::Values(
        std::make_tuple(PartialShape{16}, PartialShape{16}, 16, 16, false, PartialShape{9, 1, 2}),  // frames at 1
        std::make_tuple(PartialShape{16}, PartialShape{16}, 16, 4, false, PartialShape{9, 1, 2}),
        std::make_tuple(PartialShape{48}, PartialShape{16}, 16, 16, false, PartialShape{9, 3, 2}),
        std::make_tuple(PartialShape{8, 48}, PartialShape{16}, 16, 16, false, PartialShape{8, 9, 3, 2}),
        std::make_tuple(PartialShape{48}, PartialShape{16}, 16, 16, true, PartialShape{9, 4, 2}),
        std::make_tuple(PartialShape{5, 48}, PartialShape{16}, 16, 16, true, PartialShape{5, 9, 4, 2}),
        std::make_tuple(PartialShape{48}, PartialShape{16}, 16, 12, true, PartialShape{9, 5, 2}),
        std::make_tuple(PartialShape{56}, PartialShape{7}, 11, 3, false, PartialShape{6, 16, 2}),

        std::make_tuple(PartialShape{48}, PartialShape{8}, 16, 4, false, PartialShape{9, 9, 2}),
        std::make_tuple(PartialShape{{47, 56}}, PartialShape{7}, 11, 3, false, PartialShape{6, {13, 16}, 2}),
        std::make_tuple(PartialShape{{11, -1}}, PartialShape{7}, 11, 3, false, PartialShape{6, {1, -1}, 2}),
        std::make_tuple(PartialShape{1, 16}, PartialShape{16}, 16, 16, false, PartialShape{1, 9, 1, 2}),
        std::make_tuple(PartialShape{1, 48}, PartialShape{16}, 16, 16, false, PartialShape{1, 9, 3, 2}),
        std::make_tuple(PartialShape{2, 48}, PartialShape{8}, 16, 4, false, PartialShape{2, 9, 9, 2}),
        std::make_tuple(PartialShape{2, 9}, PartialShape{5}, 9, 100, false, PartialShape{2, 5, 1, 2}),
        std::make_tuple(PartialShape{2, 1}, PartialShape{5}, 9, 100, true, PartialShape{2, 5, 1, 2}),
        std::make_tuple(PartialShape{4, 47},
                        PartialShape{7},
                        11,
                        3,
                        false,
                        PartialShape{4, 6, 13, 2}),  // If the signal was 48, the last sample can be lost
        std::make_tuple(PartialShape{4, 56}, PartialShape{7}, 11, 3, false, PartialShape{4, 6, 16, 2}),
        std::make_tuple(PartialShape{-1, {11, -1}}, PartialShape{7}, 11, 3, false, PartialShape{-1, 6, {1, -1}, 2}),
        std::make_tuple(PartialShape{-1, {47, 56}}, PartialShape{7}, 11, 3, false, PartialShape{-1, 6, {13, 16}, 2}),
        std::make_tuple(PartialShape{{2, 4}, {11, -1}},
                        PartialShape{7},
                        11,
                        3,
                        false,
                        PartialShape{{2, 4}, 6, {1, -1}, 2}),
        std::make_tuple(PartialShape{{2, 4}, {1, -1}},
                        PartialShape{7},
                        11,
                        3,
                        true,
                        PartialShape{{2, 4}, 6, {1, -1}, 2}),

        std::make_tuple(PartialShape::dynamic(), PartialShape{7}, 11, 3, false, PartialShape::dynamic()),

        std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), 11, 3, false, PartialShape::dynamic()),
        std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic(), 11, 3, true, PartialShape::dynamic())),
    testing::PrintToStringParamName());

TEST_P(TypePropISTFTTestP, istft_shapes) {
    const auto in_data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto window = std::make_shared<Parameter>(element::f32, window_shape);
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {frame_size_val});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {step_size_val});

    const auto op = make_op(in_data, window, frame_size, frame_step, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), signal_shape);
}

TEST_F(TypePropISTFTTest, shape_length_const_out_1D) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{6, 13, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {11});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});
    const auto signal_length = Constant::create<int32_t>(element::i32, {}, {36});

    const auto op = make_op(in_data, window, frame_size, frame_step, signal_length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{36}));
}

TEST_F(TypePropISTFTTest, shape_length_const_out_2D) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{4, 6, 13, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {11});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});
    const auto signal_length = Constant::create<int32_t>(element::i32, {}, {36});

    const auto op = make_op(in_data, window, frame_size, frame_step, signal_length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 36}));
}

TEST_F(TypePropISTFTTest, shape_of_length_out_2D_with_symbols) {
    auto marked_signal = Dimension(36);
    auto symbol_s = std::make_shared<Symbol>();
    marked_signal.set_symbol(symbol_s);

    auto marked_batch = Dimension(4);
    auto symbol_b = std::make_shared<Symbol>();
    marked_batch.set_symbol(symbol_b);

    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{marked_signal});
    auto signal_length = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{marked_batch, 6, 13, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{7});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {11});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});

    const auto op = make_op(in_data, window, frame_size, frame_step, signal_length, center, normalized);
    EXPECT_EQ(op->get_output_size(), 1);

    const auto& output_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, (PartialShape{marked_batch, marked_signal}));

    EXPECT_EQ(output_shape[0].get_symbol(), symbol_b);
    EXPECT_EQ(output_shape[1].get_symbol(), symbol_s);
    EXPECT_NE(output_shape[0].get_symbol(), output_shape[1].get_symbol());
}

TEST_F(TypePropISTFTTest, data_incompatible_shape) {
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{8});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of data must be 3D or 4D"));
    }
    {
        const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of data must be 3D or 4D"));
    }
}

TEST_F(TypePropISTFTTest, window_incompatible_shape) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto window = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of window must be 1D [window_size]"));
    }
    {
        const auto window = std::make_shared<Parameter>(element::f32, PartialShape{2, 8});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of window must be 1D [window_size]"));
    }
}

TEST_F(TypePropISTFTTest, frame_size_incompatible_shape) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_size must be a scalar"));
    }
    {
        const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_size must be a scalar"));
    }
}

TEST_F(TypePropISTFTTest, frame_step_incompatible_shape) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{1});

        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_step must be a scalar"));
    }
    {
        const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("The shape of frame_step must be a scalar"));
    }
}

TEST_F(TypePropISTFTTest, signal_length_incompatible_shape) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto signal_length = std::make_shared<Parameter>(element::i64, PartialShape{1, 2});
        OV_EXPECT_THROW(
            std::ignore = make_op(in_data, window, frame_size, frame_step, signal_length, center, normalized),
            NodeValidationFailure,
            HasSubstr("The shape of 'signal_length' input must be a scalar or single element 1D tensor"));
    }
}

TEST_F(TypePropISTFTTest, data_incompatible_type) {
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto in_data = std::make_shared<Parameter>(element::i32, PartialShape{9, 3, 2});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected floating point type of the 'data' input"));
    }
    {
        const auto in_data = std::make_shared<Parameter>(element::boolean, PartialShape{9, 3, 2});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected floating point type of the 'data' input"));
    }
}

TEST_F(TypePropISTFTTest, window_incompatible_type) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto window = std::make_shared<Parameter>(element::i32, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `data` input"));
    }
    {
        const auto window = std::make_shared<Parameter>(element::boolean, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `data` input"));
    }
    {
        // Doesn't match the type of in_data
        const auto window = std::make_shared<Parameter>(element::f16, PartialShape{8});
        OV_EXPECT_THROW(
            std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
            NodeValidationFailure,
            HasSubstr("Expected floating point type of the 'window' input, matching the type of `data` input"));
    }
}

TEST_F(TypePropISTFTTest, frame_size_incompatible_type) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{8});
    const auto frame_step = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_size = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 2"));
    }
    {
        const auto frame_size = std::make_shared<Parameter>(element::i8, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 2"));
    }
}

TEST_F(TypePropISTFTTest, frame_step_incompatible_type) {
    const auto in_data = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{8});
    const auto frame_size = std::make_shared<Parameter>(element::i64, PartialShape{});
    {
        const auto frame_step = std::make_shared<Parameter>(element::f32, PartialShape{});

        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 3"));
    }
    {
        const auto frame_step = std::make_shared<Parameter>(element::i8, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(in_data, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Expected i32 or i64 type of the input at port: 3"));
    }
}

TEST_F(TypePropISTFTTest, frame_step_incompatible_value) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{8});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {16});
    {
        const auto frame_step = Constant::create<int32_t>(element::i32, {}, {-1});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Provided frame step must be greater than zero, but got: -1"));
    }
}

TEST_F(TypePropISTFTTest, frame_size_incompatible_value) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{9, 9, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{8});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {3});

    {
        const auto frame_size = Constant::create<int32_t>(element::i32, {}, {-1});
        OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, center, normalized),
                        NodeValidationFailure,
                        HasSubstr("Provided frame size must be greater than zero, but got: -1"));
    }
}

TEST_F(TypePropISTFTTest, window_incompatible_dim_with_frame_size) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{9, 9, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {8});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {4});
    OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, center, normalized),
                    NodeValidationFailure,
                    HasSubstr("Window input dimension must be in range [1, 8]"));
}

TEST_F(TypePropISTFTTest, data_shape_incompatible_dim_with_frame_size) {
    const auto signal = std::make_shared<Parameter>(element::f32, PartialShape{9, 3, 2});
    const auto window = std::make_shared<Parameter>(element::f32, PartialShape{16});
    const auto frame_size = Constant::create<int32_t>(element::i32, {}, {31});
    const auto frame_step = Constant::create<int32_t>(element::i32, {}, {11});
    OV_EXPECT_THROW(std::ignore = make_op(signal, window, frame_size, frame_step, center, normalized),
                    NodeValidationFailure,
                    HasSubstr("The dimension at data_shape[-3] must be equal to: (frame_size // 2 + 1)"));
}

}  // namespace ov::test
