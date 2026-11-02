// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

using Mode = op::internal::GatedResidual::Mode;

std::shared_ptr<op::internal::GatedResidual> make_read(const element::Type& et,
                                                        const PartialShape& residual,
                                                        const PartialShape& norm_weight,
                                                        const PartialShape& w_down,
                                                        const PartialShape& w_up,
                                                        size_t hc_count,
                                                        size_t hidden_size,
                                                        size_t lowrank,
                                                        Mode mode = Mode::READ,
                                                        bool produce_write_gate = false,
                                                        const PartialShape& w_inject = PartialShape::dynamic()) {
    auto residual_p = std::make_shared<op::v0::Parameter>(et, residual);
    auto norm_weight_p = std::make_shared<op::v0::Parameter>(et, norm_weight);
    auto w_down_p = std::make_shared<op::v0::Parameter>(et, w_down);
    auto w_up_p = std::make_shared<op::v0::Parameter>(et, w_up);

    OutputVector args{residual_p, norm_weight_p, w_down_p, w_up_p};
    if (produce_write_gate) {
        args.push_back(std::make_shared<op::v0::Parameter>(et, w_inject));
    }
    return std::make_shared<op::internal::GatedResidual>(args,
                                                         mode,
                                                         hc_count,
                                                         hidden_size,
                                                         lowrank,
                                                         1e-6F,
                                                         true,
                                                         produce_write_gate);
}

std::shared_ptr<op::internal::GatedResidual> make_write(const element::Type& et,
                                                         const PartialShape& residual,
                                                         const PartialShape& sublayer_y,
                                                         const PartialShape& write_gate,
                                                         size_t hc_count,
                                                         size_t hidden_size) {
    auto residual_p = std::make_shared<op::v0::Parameter>(et, residual);
    auto sublayer_y_p = std::make_shared<op::v0::Parameter>(et, sublayer_y);
    auto write_gate_p = std::make_shared<op::v0::Parameter>(et, write_gate);
    return std::make_shared<op::internal::GatedResidual>(OutputVector{residual_p, sublayer_y_p, write_gate_p},
                                                         Mode::WRITE,
                                                         hc_count,
                                                         hidden_size,
                                                         0,
                                                         1e-6F,
                                                         true,
                                                         false);
}

}  // namespace

TEST(type_prop, gated_residual_read_static) {
    const auto op = make_read(element::f32,
                              Shape{2, 3, 16},  // residual [B,T,C*D]
                              Shape{16},        // norm_weight
                              Shape{8, 16},     // w_down [L, C*D]
                              Shape{16, 8},     // w_up [C*D, L]
                              4,                // C
                              4,                // D
                              8);               // L

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
}

TEST(type_prop, gated_residual_read_with_write_gate) {
    const auto op = make_read(element::f16,
                              Shape{2, 3, 16},
                              Shape{16},
                              Shape{8, 16},
                              Shape{16, 8},
                              4,
                              4,
                              8,
                              Mode::READ,
                              true,
                              Shape{4, 16});

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_element_type(1), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 3, 4}));
}

TEST(type_prop, gated_residual_final_mix_static) {
    const auto op = make_read(element::bf16,
                              Shape{2, 3, 16},
                              Shape{16},
                              Shape{8, 16},
                              Shape{16, 8},
                              4,
                              4,
                              8,
                              Mode::FINAL_MIX,
                              false);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
}

TEST(type_prop, gated_residual_read_partial_shape) {
    const auto op = make_read(element::f32,
                              PartialShape{{1, 4}, -1, 16},
                              Shape{16},
                              Shape{8, 16},
                              Shape{16, 8},
                              4,
                              4,
                              8);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 4}, -1, 4}));
}

TEST(type_prop, gated_residual_write_static) {
    const auto op = make_write(element::f32,
                               Shape{2, 3, 16},  // residual [B,T,C*D]
                               Shape{2, 3, 4},   // sublayer_y [B,T,D]
                               Shape{2, 3, 4},   // write_gate [B,T,C]
                               4,
                               4);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 16}));
}

TEST(type_prop, gated_residual_read_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::f32, Shape{8, 16});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, norm_weight, w_down},
                                                                     Mode::READ,
                                                                     4,
                                                                     4,
                                                                     8,
                                                                     1e-6F,
                                                                     true,
                                                                     false),
        NodeValidationFailure,
        testing::HasSubstr("expects 4 inputs"));
}

TEST(type_prop, gated_residual_write_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, sublayer_y},
                                                                     Mode::WRITE,
                                                                     4,
                                                                     4,
                                                                     0,
                                                                     1e-6F,
                                                                     true,
                                                                     false),
        NodeValidationFailure,
        testing::HasSubstr("expects 3 inputs"));
}

TEST(type_prop, gated_residual_final_mix_cannot_produce_write_gate) {
    OV_EXPECT_THROW(std::ignore = make_read(element::f32,
                                            Shape{2, 3, 16},
                                            Shape{16},
                                            Shape{8, 16},
                                            Shape{16, 8},
                                            4,
                                            4,
                                            8,
                                            Mode::FINAL_MIX,
                                            true,
                                            Shape{4, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("FINAL_MIX"));
}

TEST(type_prop, gated_residual_invalid_hc_count) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 0, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("positive branch count"));
}

TEST(type_prop, gated_residual_invalid_hidden_size) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 0, 8),
        NodeValidationFailure,
        testing::HasSubstr("positive hidden size"));
}

TEST(type_prop, gated_residual_invalid_lowrank) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 0),
        NodeValidationFailure,
        testing::HasSubstr("positive low rank"));
}

TEST(type_prop, gated_residual_invalid_residual_rank) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("rank to be at least 2"));
}

namespace {

std::shared_ptr<op::internal::GatedResidual> make_read_compressed(bool produce_write_gate,
                                                                   bool has_zero_points,
                                                                   size_t group_size = 0) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u8, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u8, Shape{16, 8});

    OutputVector args{residual, norm_weight, w_down, w_up};
    if (produce_write_gate) {
        args.push_back(std::make_shared<op::v0::Parameter>(element::u8, Shape{4, 16}));
    }

    auto add_scale_zp = [&](size_t n) {
        args.push_back(std::make_shared<op::v0::Parameter>(element::f32, Shape{n, 1}));
        if (has_zero_points) {
            args.push_back(std::make_shared<op::v0::Parameter>(element::f32, Shape{n, 1}));
        }
    };
    add_scale_zp(8);
    add_scale_zp(16);
    if (produce_write_gate) {
        add_scale_zp(4);
    }

    return std::make_shared<op::internal::GatedResidual>(args,
                                                          Mode::READ,
                                                          4,
                                                          4,
                                                          8,
                                                          1e-6F,
                                                          true,
                                                          produce_write_gate,
                                                          element::dynamic,
                                                          true,
                                                          has_zero_points,
                                                          group_size);
}

}  // namespace

TEST(type_prop, gated_residual_compressed_weights_no_zero_points) {
    const auto op = make_read_compressed(false, false);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
}

TEST(type_prop, gated_residual_compressed_weights_with_zero_points) {
    const auto op = make_read_compressed(false, true);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
}

TEST(type_prop, gated_residual_compressed_weights_with_write_gate) {
    const auto op = make_read_compressed(true, true);
    EXPECT_EQ(op->get_output_size(), 2);
}

TEST(type_prop, gated_residual_compressed_weights_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u8, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u8, Shape{16, 8});
    auto w_down_scale = std::make_shared<op::v0::Parameter>(element::f32, Shape{8, 1});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up, w_down_scale},
                        Mode::READ,
                        4,
                        4,
                        8,
                        1e-6F,
                        true,
                        false,
                        element::dynamic,
                        true,
                        false,
                        0),
                    NodeValidationFailure,
                    testing::HasSubstr("expects"));
}

TEST(type_prop, gated_residual_compressed_weights_invalid_group_size) {
    OV_EXPECT_THROW(std::ignore = make_read_compressed(false, false, 2),
                    NodeValidationFailure,
                    testing::HasSubstr("group size"));
}

TEST(type_prop, gated_residual_write_rejects_compressed_weights) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto write_gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, sublayer_y, write_gate},
                        Mode::WRITE,
                        4,
                        4,
                        0,
                        1e-6F,
                        true,
                        false,
                        element::dynamic,
                        true,
                        false,
                        0),
                    NodeValidationFailure,
                    testing::HasSubstr("does not support compressed weights"));
}

}  // namespace ov::test
