// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <sstream>
#include <string>
#include <tuple>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {

class TypePropGatedResidualTest : public TypePropOpTest<op::internal::GatedResidual> {};

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
    return std::make_shared<op::internal::GatedResidual>(
        args,
        ov::op::internal::GatedResidual::Config{mode, hc_count, hidden_size, lowrank, 1e-6F, produce_write_gate});
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
    return std::make_shared<op::internal::GatedResidual>(
        OutputVector{residual_p, sublayer_y_p, write_gate_p},
        ov::op::internal::GatedResidual::Config{Mode::WRITE, hc_count, hidden_size, 0, 1e-6F, false});
}

}  // namespace

TEST_F(TypePropGatedResidualTest, read_static) {
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

TEST_F(TypePropGatedResidualTest, read_with_write_gate) {
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

TEST_F(TypePropGatedResidualTest, final_mix_static) {
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

TEST_F(TypePropGatedResidualTest, read_partial_shape) {
    const auto op =
        make_read(element::f32, PartialShape{{1, 4}, -1, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 4}, -1, 4}));
}

TEST_F(TypePropGatedResidualTest, write_static) {
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

TEST_F(TypePropGatedResidualTest, read_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::f32, Shape{8, 16});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down},
                        ov::op::internal::GatedResidual::Config{Mode::READ, 4, 4, 8, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("expects 4 inputs"));
}

TEST_F(TypePropGatedResidualTest, write_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, sublayer_y},
                        ov::op::internal::GatedResidual::Config{Mode::WRITE, 4, 4, 0, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("expects 3 inputs"));
}

TEST_F(TypePropGatedResidualTest, final_mix_cannot_produce_write_gate) {
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

TEST_F(TypePropGatedResidualTest, invalid_hc_count) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 0, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("positive branch count"));
}

TEST_F(TypePropGatedResidualTest, invalid_hidden_size) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 0, 8),
        NodeValidationFailure,
        testing::HasSubstr("positive hidden size"));
}

TEST_F(TypePropGatedResidualTest, invalid_lowrank) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 0),
        NodeValidationFailure,
        testing::HasSubstr("positive low rank"));
}

TEST_F(TypePropGatedResidualTest, invalid_residual_rank) {
    OV_EXPECT_THROW(std::ignore = make_read(element::f32, Shape{16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
                    NodeValidationFailure,
                    testing::HasSubstr("rank to be at least 2"));
}

namespace {

// C = D = 4 -> C*D = 16, L = 8. Scales / zero points are [N, G] with the group size along K being K / G.
struct CompressedCfg {
    element::Type weight_et = element::u4;
    element::Type scale_et = element::f16;
    element::Type zp_et = element::dynamic;  // dynamic -> same as the weights
    bool has_zp = true;
    bool write_gate = false;
    size_t g_down = 4;                 // K = 16
    size_t g_up = 2;                   // K = 8
    size_t g_inject = 4;               // K = 16
    std::optional<Shape> scale_shape;  // replaces the [N, G] shape of every scale
    std::optional<Shape> zp_shape;     // replaces the [N, G] shape of every zero point
    Mode mode = Mode::READ;
};

std::shared_ptr<op::internal::GatedResidual> make_read_compressed(const CompressedCfg& c) {
    const auto zp_et = c.zp_et.is_dynamic() ? c.weight_et : c.zp_et;
    auto param = [](const element::Type& et, const Shape& shape) {
        return std::make_shared<op::v0::Parameter>(et, shape);
    };
    const auto scale = [&](size_t n, size_t g) {
        return param(c.scale_et, c.scale_shape.value_or(Shape{n, g}));
    };
    const auto zp = [&](size_t n, size_t g) {
        return param(zp_et, c.zp_shape.value_or(Shape{n, g}));
    };
    OutputVector args{param(element::f32, Shape{2, 3, 16}),
                      param(element::f32, Shape{16}),
                      param(c.weight_et, Shape{8, 16}),
                      param(c.weight_et, Shape{16, 8}),
                      scale(8, c.g_down),
                      scale(16, c.g_up)};
    if (c.has_zp) {
        args.push_back(zp(8, c.g_down));
        args.push_back(zp(16, c.g_up));
    }
    if (c.write_gate) {
        args.push_back(param(c.weight_et, Shape{4, 16}));
        args.push_back(scale(4, c.g_inject));
        if (c.has_zp) {
            args.push_back(zp(4, c.g_inject));
        }
    }
    return std::make_shared<op::internal::GatedResidual>(
        args,
        ov::op::internal::GatedResidual::Config{c.mode, 4, 4, 8, 1e-6F, c.write_gate});
}

}  // namespace

using WeightTypeParams = std::tuple<element::Type, bool>;  // weight type, zero points

class TypePropGatedResidualWeightTypeTest : public TypePropGatedResidualTest,
                                            public testing::WithParamInterface<WeightTypeParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<WeightTypeParams>& info) {
        const auto& [et, has_zp] = info.param;
        return et.to_string() + (has_zp ? "_zero_points" : "_no_zero_points");
    }
};

TEST_P(TypePropGatedResidualWeightTypeTest, compressed_weight_types) {
    const auto& [et, has_zp] = GetParam();
    CompressedCfg cfg;
    cfg.weight_et = et;
    cfg.has_zp = has_zp;
    const auto op = make_read_compressed(cfg);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
    const auto layout = op->get_read_layout();
    ASSERT_TRUE(layout.has_value());
    EXPECT_TRUE(layout->compressed);
    EXPECT_EQ(layout->has_zp, has_zp);
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualWeightTypeTest,
                         testing::Combine(testing::Values(element::i4, element::u4, element::i8, element::u8),
                                          testing::Bool()),
                         TypePropGatedResidualWeightTypeTest::get_test_name);

class TypePropGatedResidualZeroPointsTest : public TypePropGatedResidualTest, public testing::WithParamInterface<bool> {
public:
    static std::string get_test_name(const testing::TestParamInfo<bool>& info) {
        return info.param ? "zero_points" : "no_zero_points";
    }
};

TEST_P(TypePropGatedResidualZeroPointsTest, compressed_with_write_gate) {
    const bool has_zp = GetParam();
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

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualZeroPointsTest,
                         testing::Bool(),
                         TypePropGatedResidualZeroPointsTest::get_test_name);

TEST_F(TypePropGatedResidualTest, compressed_final_mix) {
    CompressedCfg cfg;
    cfg.mode = Mode::FINAL_MIX;
    const auto op = make_read_compressed(cfg);
    EXPECT_EQ(op->get_output_size(), 1);
}

class TypePropGatedResidualWriteGateTest : public TypePropGatedResidualTest, public testing::WithParamInterface<bool> {
public:
    static std::string get_test_name(const testing::TestParamInfo<bool>& info) {
        return info.param ? "write_gate" : "no_write_gate";
    }
};

TEST_P(TypePropGatedResidualWriteGateTest, plain_read_layout) {
    const bool write_gate = GetParam();
    const auto plain = make_read(element::f32,
                                 Shape{2, 3, 16},
                                 Shape{16},
                                 Shape{8, 16},
                                 Shape{16, 8},
                                 4,
                                 4,
                                 8,
                                 Mode::READ,
                                 write_gate,
                                 Shape{4, 16});
    const auto layout = plain->get_read_layout();
    ASSERT_TRUE(layout.has_value());
    EXPECT_FALSE(layout->compressed);
    EXPECT_EQ(layout->write_gate, write_gate);
    EXPECT_EQ(layout->input_count, plain->get_input_size());
    // The positions the plugins read: every form starts with residual, norm, w_down, w_up.
    EXPECT_EQ(layout->residual, 0U);
    EXPECT_EQ(layout->norm_weight, 1U);
    EXPECT_EQ(layout->w_down, 2U);
    EXPECT_EQ(layout->w_up, 3U);
    EXPECT_EQ(layout->w_inject, 4U);
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualWriteGateTest,
                         testing::Bool(),
                         TypePropGatedResidualWriteGateTest::get_test_name);

using CompressedLayoutParams = std::tuple<bool, bool>;  // write gate, zero points

class TypePropGatedResidualCompressedLayoutTest : public TypePropGatedResidualTest,
                                                  public testing::WithParamInterface<CompressedLayoutParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<CompressedLayoutParams>& info) {
        const auto& [write_gate, has_zp] = info.param;
        return std::string(write_gate ? "write_gate" : "no_write_gate") + (has_zp ? "_zero_points" : "_no_zero_points");
    }
};

TEST_P(TypePropGatedResidualCompressedLayoutTest, compressed_read_layout) {
    const auto& [write_gate, has_zp] = GetParam();
    CompressedCfg cfg;
    cfg.write_gate = write_gate;
    cfg.has_zp = has_zp;
    const auto compressed = make_read_compressed(cfg);
    const auto layout = compressed->get_read_layout();
    ASSERT_TRUE(layout.has_value());
    EXPECT_TRUE(layout->compressed);
    EXPECT_EQ(layout->has_zp, has_zp);
    EXPECT_EQ(layout->write_gate, write_gate);
    EXPECT_EQ(layout->input_count, compressed->get_input_size());
    EXPECT_EQ(layout->scale_down, 4U);
    EXPECT_EQ(layout->scale_up, 5U);
    if (has_zp) {
        EXPECT_EQ(layout->zp_down, 6U);
        EXPECT_EQ(layout->zp_up, 7U);
    }
    EXPECT_EQ(layout->w_inject, has_zp ? 8U : 6U);
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualCompressedLayoutTest,
                         testing::Combine(testing::Bool(), testing::Bool()),
                         TypePropGatedResidualCompressedLayoutTest::get_test_name);

TEST_F(TypePropGatedResidualTest, compressed_wrong_input_count) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u4, Shape{16, 8});
    auto w_down_scale = std::make_shared<op::v0::Parameter>(element::f16, Shape{8, 4});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up, w_down_scale},
                        ov::op::internal::GatedResidual::Config{Mode::READ, 4, 4, 8, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("expects"));
}

class TypePropGatedResidualScaleTypeTest : public TypePropGatedResidualTest,
                                           public testing::WithParamInterface<element::Type> {
public:
    static std::string get_test_name(const testing::TestParamInfo<element::Type>& info) {
        return info.param.to_string();
    }
};

TEST_P(TypePropGatedResidualScaleTypeTest, compressed_scale_floating_point_types) {
    CompressedCfg cfg;
    cfg.scale_et = GetParam();
    EXPECT_NO_THROW(std::ignore = make_read_compressed(cfg));
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualScaleTypeTest,
                         testing::Values(element::f16, element::bf16, element::f32),
                         TypePropGatedResidualScaleTypeTest::get_test_name);

TEST_F(TypePropGatedResidualTest, compressed_scale_must_be_floating_point) {
    CompressedCfg cfg;
    cfg.scale_et = element::i32;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("scale must be a floating-point type"));
}

TEST_F(TypePropGatedResidualTest, compressed_zero_point_must_match_weight_type) {
    CompressedCfg cfg;
    cfg.weight_et = element::u4;
    cfg.zp_et = element::u8;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("zero point must have the weight element type"));
}

TEST_F(TypePropGatedResidualTest, compressed_group_count_must_divide_k) {
    CompressedCfg cfg;
    cfg.g_up = 3;  // K = 8 is not divisible by 3
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("not divisible by the scale group count"));
}

TEST_F(TypePropGatedResidualTest, compressed_scale_shape_is_n_by_g) {
    // [G, N] instead of [N, G].
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::u4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::u4, Shape{16, 8});
    auto scale_down = std::make_shared<op::v0::Parameter>(element::f16, Shape{4, 8});
    auto scale_up = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 16});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up, scale_down, scale_up},
                        ov::op::internal::GatedResidual::Config{Mode::READ, 4, 4, 8, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("scale must have shape [8, G]"));
}

using SingleElementParams = std::tuple<Shape, bool>;  // shape of every scale and zero point, write gate

class TypePropGatedResidualSingleElementTest : public TypePropGatedResidualTest,
                                               public testing::WithParamInterface<SingleElementParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<SingleElementParams>& info) {
        const auto& [shape, write_gate] = info.param;
        std::ostringstream name;
        name << "rank" << shape.size() << (write_gate ? "_write_gate" : "_no_write_gate");
        return name.str();
    }
};

// A single-element scale or zero point (per-tensor) is accepted, as by FullyConnectedCompressed.
TEST_P(TypePropGatedResidualSingleElementTest, compressed_single_element_scale_and_zero_point) {
    const auto& [shape, write_gate] = GetParam();
    CompressedCfg cfg;
    cfg.write_gate = write_gate;
    cfg.scale_shape = shape;
    cfg.zp_shape = shape;
    const auto op = make_read_compressed(cfg);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 3, 4}));
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         TypePropGatedResidualSingleElementTest,
                         testing::Combine(testing::Values(Shape{}, Shape{1}, Shape{1, 1}), testing::Bool()),
                         TypePropGatedResidualSingleElementTest::get_test_name);

TEST_F(TypePropGatedResidualTest, compressed_single_element_zero_point_with_full_scale) {
    CompressedCfg cfg;
    cfg.zp_shape = Shape{1};
    EXPECT_NO_THROW(std::ignore = make_read_compressed(cfg));
}

TEST_F(TypePropGatedResidualTest, compressed_scale_with_several_elements_must_be_n_by_g) {
    CompressedCfg cfg;
    cfg.scale_shape = Shape{2};
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("scale must have shape [8, G] or one element"));
}

// The zero point is a single element or has the shape of the scale; a single-element scale with full zero points is not
// a form the backend runs.
TEST_F(TypePropGatedResidualTest, compressed_zero_point_shape_is_scale_shape_or_single_element) {
    CompressedCfg cfg;
    cfg.scale_shape = Shape{1};
    cfg.zp_shape = Shape{8, 2};
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("zero point must match the scale shape or be one element"));
}

TEST_F(TypePropGatedResidualTest, quantized_weights_need_compressed_form) {
    // i4 weights with only the 4 uncompressed inputs.
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::i4, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::i4, Shape{16, 8});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up},
                        ov::op::internal::GatedResidual::Config{Mode::READ, 4, 4, 8, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("exactly with the compressed input forms"));
}

TEST_F(TypePropGatedResidualTest, float_weights_with_compressed_form) {
    CompressedCfg cfg;
    cfg.weight_et = element::f32;
    OV_EXPECT_THROW(std::ignore = make_read_compressed(cfg),
                    NodeValidationFailure,
                    testing::HasSubstr("exactly with the compressed input forms"));
}

TEST_F(TypePropGatedResidualTest, weights_must_share_element_type) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto norm_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{16});
    auto w_down = std::make_shared<op::v0::Parameter>(element::f32, Shape{8, 16});
    auto w_up = std::make_shared<op::v0::Parameter>(element::f16, Shape{16, 8});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, norm_weight, w_down, w_up},
                        ov::op::internal::GatedResidual::Config{Mode::READ, 4, 4, 8, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("must share one element type"));
}

TEST_F(TypePropGatedResidualTest, wrong_weight_shape) {
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

TEST_F(TypePropGatedResidualTest, residual_last_dim_must_be_c_times_d) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::f32, Shape{2, 3, 12}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("last dimension must be C*D"));
}

TEST_F(TypePropGatedResidualTest, residual_must_be_floating_point) {
    OV_EXPECT_THROW(
        std::ignore = make_read(element::i32, Shape{2, 3, 16}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
        NodeValidationFailure,
        testing::HasSubstr("floating-point"));
}

TEST_F(TypePropGatedResidualTest, write_inputs_must_share_element_type) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 16});
    auto sublayer_y = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 3, 4});
    auto write_gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GatedResidual>(
                        OutputVector{residual, sublayer_y, write_gate},
                        ov::op::internal::GatedResidual::Config{Mode::WRITE, 4, 4, 0, 1e-6F, false}),
                    NodeValidationFailure,
                    testing::HasSubstr("share the residual element type"));
}

TEST_F(TypePropGatedResidualTest, write_dynamic_residual_width_is_inferred) {
    // The residual's width is C*D, and its dynamic leading dimensions are taken from Y and the gate.
    const auto op = make_write(element::f32, PartialShape{-1, -1, -1}, Shape{2, 3, 4}, Shape{2, 3, 4}, 4, 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 16}));
}

TEST_F(TypePropGatedResidualTest, write_leading_dims_are_merged) {
    const auto op =
        make_write(element::f32, PartialShape{-1, 3, 16}, PartialShape{2, -1, 4}, PartialShape{-1, -1, 4}, 4, 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 16}));
}

TEST_F(TypePropGatedResidualTest, write_dynamic_rank_inputs_are_accepted) {
    EXPECT_NO_THROW(std::ignore =
                        make_write(element::f32, PartialShape::dynamic(), Shape{2, 3, 4}, Shape{2, 3, 4}, 4, 4));
    EXPECT_NO_THROW(
        std::ignore =
            make_write(element::f32, Shape{2, 3, 16}, PartialShape::dynamic(), PartialShape::dynamic(), 4, 4));
}

TEST_F(TypePropGatedResidualTest, write_mismatched_leading_dims) {
    // Y and the gate with fewer rows than the residual would be read out of bounds by the kernel.
    OV_EXPECT_THROW(std::ignore = make_write(element::f32, Shape{2, 3, 16}, Shape{1, 1, 4}, Shape{1, 1, 4}, 4, 4),
                    NodeValidationFailure,
                    testing::HasSubstr("matching leading dimensions"));
    OV_EXPECT_THROW(std::ignore = make_write(element::f32, Shape{6, 16}, Shape{6, 4}, Shape{1, 4}, 4, 4),
                    NodeValidationFailure,
                    testing::HasSubstr("matching leading dimensions"));
    OV_EXPECT_THROW(std::ignore = make_write(element::f32, Shape{2, 3, 16}, Shape{2, 4, 4}, Shape{2, 3, 4}, 4, 4),
                    NodeValidationFailure,
                    testing::HasSubstr("matching leading dimensions"));
}

TEST_F(TypePropGatedResidualTest, write_mismatched_rank) {
    OV_EXPECT_THROW(std::ignore = make_write(element::f32, Shape{2, 3, 16}, Shape{6, 4}, Shape{2, 3, 4}, 4, 4),
                    NodeValidationFailure,
                    testing::HasSubstr("must have the same rank"));
    OV_EXPECT_THROW(std::ignore = make_write(element::f32, Shape{2, 3, 16}, Shape{2, 3, 4}, Shape{4}, 4, 4),
                    NodeValidationFailure,
                    testing::HasSubstr("must have the same rank"));
}

TEST_F(TypePropGatedResidualTest, unknown_mode_is_rejected) {
    OV_EXPECT_THROW(std::ignore = make_read(element::f32,
                                            Shape{2, 3, 16},
                                            Shape{16},
                                            Shape{8, 16},
                                            Shape{16, 8},
                                            4,
                                            4,
                                            8,
                                            // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
                                            static_cast<Mode>(7)),
                    NodeValidationFailure,
                    testing::HasSubstr("unknown mode"));
}

TEST_F(TypePropGatedResidualTest, c_times_d_size_t_overflow_is_rejected) {
    // C * D = 2^64 wraps to 0 in size_t.
    OV_EXPECT_THROW(std::ignore = make_read(element::f32,
                                            PartialShape::dynamic(),
                                            PartialShape::dynamic(),
                                            Shape{8, 16},
                                            Shape{16, 8},
                                            size_t{1} << 32,
                                            size_t{1} << 32,
                                            8),
                    NodeValidationFailure,
                    testing::HasSubstr("overflows"));
}

TEST_F(TypePropGatedResidualTest, c_times_d_int64_overflow_is_rejected) {
    // C * D = 2^63 fits in size_t but not in the int64 dimensions of the shape system.
    OV_EXPECT_THROW(std::ignore = make_read(element::f32,
                                            PartialShape::dynamic(),
                                            PartialShape::dynamic(),
                                            Shape{8, 16},
                                            Shape{16, 8},
                                            2,
                                            size_t{1} << 62,
                                            8),
                    NodeValidationFailure,
                    testing::HasSubstr("overflows"));
}

TEST_F(TypePropGatedResidualTest, read_dynamic_rank) {
    const auto op = make_read(element::f32, PartialShape::dynamic(), Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropGatedResidualTest, read_with_write_gate_dynamic_rank) {
    const auto op = make_read(element::f32,
                              PartialShape::dynamic(),
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
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape::dynamic());
}

TEST_F(TypePropGatedResidualTest, read_interval_last_dimension_is_replaced) {
    const auto op =
        make_read(element::f32, PartialShape{{2, 5}, {1, 4}, {10, 20}}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{2, 5}, {1, 4}, 4}));
}

TEST_F(TypePropGatedResidualTest, read_symbols_of_leading_dimensions_are_kept) {
    auto residual = PartialShape{-1, -1, 16};
    const auto symbols = set_shape_symbols(residual);
    const auto op = make_read(element::f32,
                              residual,
                              Shape{16},
                              Shape{8, 16},
                              Shape{16, 8},
                              4,
                              4,
                              8,
                              Mode::READ,
                              true,
                              Shape{4, 16});

    const TensorSymbol expected{symbols[0], symbols[1], nullptr};
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), expected);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(1)), expected);
}

TEST_F(TypePropGatedResidualTest, write_interval_dimensions_are_intersected) {
    const auto op = make_write(element::f32,
                               PartialShape{{1, 4}, -1, 16},
                               PartialShape{{2, 6}, -1, 4},
                               PartialShape{-1, {3, 5}, 4},
                               4,
                               4);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{2, 4}, {3, 5}, 16}));
}

TEST_F(TypePropGatedResidualTest, write_symbols_of_leading_dimensions_are_kept) {
    auto residual = PartialShape{-1, -1, 16};
    const auto symbols = set_shape_symbols(residual);
    auto sublayer_y = PartialShape{-1, -1, 4};
    auto write_gate = PartialShape{-1, -1, 4};
    const TensorSymbol leading{symbols[0], symbols[1], nullptr};
    set_shape_symbols(sublayer_y, leading);
    set_shape_symbols(write_gate, leading);
    const auto op = make_write(element::f32, residual, sublayer_y, write_gate, 4, 4);

    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), leading);
}

TEST_F(TypePropGatedResidualTest, invalid_residual_scalar) {
    OV_EXPECT_THROW(std::ignore = make_read(element::f32, Shape{}, Shape{16}, Shape{8, 16}, Shape{16, 8}, 4, 4, 8),
                    NodeValidationFailure,
                    testing::HasSubstr("rank to be at least 2"));
}

}  // namespace ov::test
