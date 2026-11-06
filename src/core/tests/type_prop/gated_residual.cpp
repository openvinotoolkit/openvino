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
    const auto op =
        make_read(element::f32, PartialShape{{1, 4}, -1, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8);

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

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, sublayer_y},
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
    OV_EXPECT_THROW(std::ignore = make_read(element::f32, Shape{16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
                    NodeValidationFailure,
                    testing::HasSubstr("rank to be at least 2"));
}

namespace {

// C = D = 4 -> C*D = 16, L = 8. Scales / zero points are [G, N] with the group size along K being K / G.
struct CompressedCfg {
    element::Type weight_et = element::u4;
    element::Type scale_et = element::f16;
    element::Type zp_et = element::dynamic;  // dynamic -> same as the weights
    bool has_zp = true;
    bool write_gate = false;
    size_t g_down = 4;    // K = 16
    size_t g_up = 2;      // K = 8
    size_t g_inject = 4;  // K = 16
    Mode mode = Mode::READ;
};

std::shared_ptr<op::internal::GatedResidual> make_read_compressed(const CompressedCfg& c) {
    const auto zp_et = c.zp_et.is_dynamic() ? c.weight_et : c.zp_et;
    auto param = [](const element::Type& et, const Shape& shape) {
        return std::make_shared<op::v0::Parameter>(et, shape);
    };
    OutputVector args{param(element::f32, Shape{2, 3, 16}),
                      param(element::f32, Shape{16}),
                      param(c.weight_et, Shape{8, 16}),
                      param(c.weight_et, Shape{16, 8}),
                      param(c.scale_et, Shape{c.g_down, 8}),
                      param(c.scale_et, Shape{c.g_up, 16})};
    if (c.has_zp) {
        args.push_back(param(zp_et, Shape{c.g_down, 8}));
        args.push_back(param(zp_et, Shape{c.g_up, 16}));
    }
    if (c.write_gate) {
        args.push_back(param(c.weight_et, Shape{4, 16}));
        args.push_back(param(c.scale_et, Shape{c.g_inject, 4}));
        if (c.has_zp) {
            args.push_back(param(zp_et, Shape{c.g_inject, 4}));
        }
    }
    return std::make_shared<op::internal::GatedResidual>(args, c.mode, 4, 4, 8, 1e-6F, true, c.write_gate);
}

}  // namespace

TEST(type_prop, gated_residual_compressed_weight_types) {
    for (const auto& et : {element::i4, element::u4, element::i8, element::u8}) {
        for (const bool has_zp : {false, true}) {
            CompressedCfg cfg;
            cfg.weight_et = et;
            cfg.has_zp = has_zp;
            const auto op = make_read_compressed(cfg);
            EXPECT_EQ(op->get_output_size(), 1) << et << " zp=" << has_zp;
            EXPECT_EQ(op->get_output_element_type(0), element::f32);
            EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
            const auto layout = op->get_read_layout();
            ASSERT_TRUE(layout.has_value());
            EXPECT_TRUE(layout->compressed);
            EXPECT_EQ(layout->has_zp, has_zp);
        }
    }
}

TEST(type_prop, gated_residual_compressed_with_write_gate) {
    for (const bool has_zp : {false, true}) {
        CompressedCfg cfg;
        cfg.write_gate = true;
        cfg.has_zp = has_zp;
        const auto op = make_read_compressed(cfg);
        EXPECT_EQ(op->get_output_size(), 2);
        EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 3, 4}));
        // The write-gate inputs are always the last ones: 8 without / 11 with zero points.
        EXPECT_EQ(op->get_input_size(), has_zp ? 11U : 8U);
        const auto layout = op->get_read_layout();
        ASSERT_TRUE(layout.has_value());
        EXPECT_EQ(layout->w_inject, has_zp ? 8U : 6U);
        EXPECT_EQ(layout->scale_inject, layout->w_inject + 1);
    }
}

TEST(type_prop, gated_residual_compressed_final_mix) {
    CompressedCfg cfg;
    cfg.mode = Mode::FINAL_MIX;
    const auto op = make_read_compressed(cfg);
    EXPECT_EQ(op->get_output_size(), 1);
}

TEST(type_prop, gated_residual_read_layout_by_input_count) {
    using Op = op::internal::GatedResidual;
    EXPECT_EQ(Op::read_layout(4, false)->input_count, 4U);
    EXPECT_EQ(Op::read_layout(6, false)->has_zp, false);
    EXPECT_EQ(Op::read_layout(8, false)->has_zp, true);
    EXPECT_EQ(Op::read_layout(5, true)->compressed, false);
    EXPECT_EQ(Op::read_layout(8, true)->has_zp, false);
    EXPECT_EQ(Op::read_layout(11, true)->has_zp, true);
    for (const size_t n : {0U, 1U, 3U, 5U, 7U, 9U, 10U, 12U}) {
        EXPECT_FALSE(Op::read_layout(n, false).has_value()) << n;
    }
    for (const size_t n : {4U, 6U, 7U, 9U, 10U, 12U}) {
        EXPECT_FALSE(Op::read_layout(n, true).has_value()) << n;
    }
}

TEST(type_prop, gated_residual_compressed_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u4, Shape{16, 8});
    auto w_down_scale = std::make_shared<op::v0::Parameter>(element::f16, Shape{4, 8});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up, w_down_scale},
                        Mode::READ,
                        4,
                        4,
                        8,
                        1e-6F,
                        true,
                        false),
                    NodeValidationFailure,
                    testing::HasSubstr("expects"));
}

TEST(type_prop, gated_residual_compressed_scale_floating_point_types) {
    for (const auto& et : {element::f16, element::bf16, element::f32}) {
        CompressedCfg cfg;
        cfg.scale_et = et;
        EXPECT_NO_THROW(std::ignore = make_read_compressed(cfg)) << et;
    }
}

TEST(type_prop, gated_residual_compressed_scale_must_be_floating_point) {
    CompressedCfg cfg;
    cfg.scale_et = element::i32;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("scale must be a floating-point type"));
}

TEST(type_prop, gated_residual_compressed_zero_point_must_match_weight_type) {
    CompressedCfg cfg;
    cfg.weight_et = element::u4;
    cfg.zp_et = element::u8;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("zero point must have the weight element type"));
}

TEST(type_prop, gated_residual_compressed_group_count_must_divide_k) {
    CompressedCfg cfg;
    cfg.g_up = 3;  // K = 8 is not divisible by 3
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("not divisible by the scale group count"));
}

TEST(type_prop, gated_residual_compressed_scale_shape_is_group_major) {
    // [N, G] (the FC layout) instead of [G, N].
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u4, Shape{16, 8});
    auto scale_down = std::make_shared<op::v0::Parameter>(element::f16, Shape{8, 4});
    auto scale_up = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 16});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up, scale_down, scale_up},
                        Mode::READ,
                        4,
                        4,
                        8,
                        1e-6F,
                        true,
                        false),
                    NodeValidationFailure,
                    testing::HasSubstr("scale must have shape [G, 8]"));
}

TEST(type_prop, gated_residual_quantized_weights_need_compressed_form) {
    // i4 weights with only the 4 uncompressed inputs.
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::i4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::i4, Shape{16, 8});
    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, norm_weight, w_down, w_up},
                                                                    Mode::READ,
                                                                    4,
                                                                    4,
                                                                    8,
                                                                    1e-6F,
                                                                    true,
                                                                    false),
        NodeValidationFailure,
        testing::HasSubstr("exactly with the compressed input forms"));
}

TEST(type_prop, gated_residual_float_weights_with_compressed_form) {
    CompressedCfg cfg;
    cfg.weight_et = element::f32;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("exactly with the compressed input forms"));
}

TEST(type_prop, gated_residual_weights_must_share_element_type) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::f32, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::f16, Shape{16, 8});
    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, norm_weight, w_down, w_up},
                                                                    Mode::READ,
                                                                    4,
                                                                    4,
                                                                    8,
                                                                    1e-6F,
                                                                    true,
                                                                    false),
        NodeValidationFailure,
        testing::HasSubstr("must share one element type"));
}

TEST(type_prop, gated_residual_wrong_weight_shape) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{16, 8}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("W_down must have shape"));
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{8, 16}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("W_up must have shape"));
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{8}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("norm weight must have shape"));
}

TEST(type_prop, gated_residual_residual_last_dim_must_be_c_times_d) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 12}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("last dimension must be C*D"));
}

TEST(type_prop, gated_residual_residual_must_be_floating_point) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::i32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("floating-point"));
}

TEST(type_prop, gated_residual_write_inputs_must_share_element_type) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 3, 4});
    auto write_gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::GatedResidual>(OutputVector{residual, sublayer_y, write_gate},
                                                                    Mode::WRITE,
                                                                    4,
                                                                    4,
                                                                    0,
                                                                    1e-6F,
                                                                    true,
                                                                    false),
        NodeValidationFailure,
        testing::HasSubstr("share the residual element type"));
}

TEST(type_prop, gated_residual_write_dynamic_residual_width_is_inferred) {
    const auto op = make_write(element::f32, PartialShape{-1, -1, -1}, Shape{2, 3, 4}, Shape{2, 3, 4}, 4, 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1, 16}));
}

}  // namespace ov::test
