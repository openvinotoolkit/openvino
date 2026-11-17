// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/restore_u2_gather_dequant.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

namespace {

using namespace ov;

constexpr size_t VOCAB = 5;
constexpr size_t GROUPS = 2;
constexpr size_t PACKED_GROUP_SIZE = 256;                  // bytes, four 2-bit codes each
constexpr size_t HIDDEN = GROUPS * PACKED_GROUP_SIZE * 4;  // the host gather requires the embedding size >= 2048
constexpr size_t SMALL_PACKED_GROUP_SIZE = 2;

uint8_t packed_byte(size_t index) {
    return static_cast<uint8_t>((index * 37 + 11) & 0xFF);
}

float scale_value(size_t index) {
    return 0.25f * static_cast<float>(index + 1);
}

std::vector<uint8_t> packed_bytes(const size_t packed_group_size) {
    std::vector<uint8_t> bytes(VOCAB * GROUPS * packed_group_size);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = packed_byte(i);
    }
    return bytes;
}

std::shared_ptr<op::v0::Constant> make_scales() {
    std::vector<float16> scales(VOCAB * GROUPS);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = float16(scale_value(i));
    }
    return op::v0::Constant::create(element::f16, Shape{VOCAB, GROUPS, 1}, scales);
}

// The zero point is a f16 Constant, or Convert(Constant u8), the form the GGUF frontend emits.
std::shared_ptr<Node> make_zero_point(const float zero_point, const bool u8_zero_point) {
    if (u8_zero_point) {
        return std::make_shared<op::v0::Convert>(
            op::v0::Constant::create(element::u8, Shape{}, {static_cast<uint8_t>(zero_point)}),
            element::f16);
    }
    return op::v0::Constant::create(element::f16, Shape{}, {zero_point});
}

struct PackedModelParams {
    float zero_point = 1.f;
    int32_t shift_step = 2;
    // Shifts along the axis before the packed bytes, which orders the codes differently from a u2 Constant.
    bool code_major = false;
    bool u8_zero_point = false;
    size_t packed_group_size = PACKED_GROUP_SIZE;
};

// Mirrors the packed gather of the GGUF frontend: Gather over u8 rows, then the bitwise unpacking of
// the 2-bit codes, the zero point, and Gather over the scales.
std::shared_ptr<Model> build_packed_gather_model(const PackedModelParams& p = {}) {
    const auto group_size = static_cast<int64_t>(p.packed_group_size);
    const auto packed = op::v0::Constant::create(element::u8,
                                                 Shape{VOCAB, GROUPS, p.packed_group_size},
                                                 packed_bytes(p.packed_group_size));
    const auto ids = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    const auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    const auto selected = std::make_shared<op::v8::Gather>(packed, ids, axis);
    const auto widened = std::make_shared<op::v0::Convert>(selected, element::i32);
    const auto expanded = std::make_shared<op::v1::Reshape>(
        widened,
        op::v0::Constant::create(element::i64,
                                 Shape{5},
                                 p.code_major ? std::vector<int64_t>{0, 0, GROUPS, 1, group_size}
                                              : std::vector<int64_t>{0, 0, GROUPS, group_size, 1}),
        true);
    const auto shifted = std::make_shared<op::v15::BitwiseRightShift>(
        expanded,
        op::v0::Constant::create(element::i32,
                                 p.code_major ? Shape{4, 1} : Shape{4},
                                 std::vector<int32_t>{0, p.shift_step, 2 * p.shift_step, 3 * p.shift_step}));
    const auto codes =
        std::make_shared<op::v13::BitwiseAnd>(shifted, op::v0::Constant::create(element::i32, Shape{}, {3}));
    const auto grouped = std::make_shared<op::v1::Reshape>(
        codes,
        op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 0, GROUPS, group_size * 4}),
        true);
    const auto values = std::make_shared<op::v0::Convert>(grouped, element::f16);
    const auto centered = std::make_shared<op::v1::Subtract>(values, make_zero_point(p.zero_point, p.u8_zero_point));
    const auto selected_scales = std::make_shared<op::v8::Gather>(make_scales(), ids, axis);
    const auto dequantized = std::make_shared<op::v1::Multiply>(centered, selected_scales);
    const auto restored = std::make_shared<op::v1::Reshape>(
        dequantized,
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{0, 0, static_cast<int64_t>(GROUPS) * group_size * 4}),
        true);
    const auto embeds = std::make_shared<op::v0::Convert>(restored, element::f32);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(embeds)}, ParameterVector{ids});
}

// Gather over the dequantized u2 vocabulary, which is what the pass rewrites build_packed_gather_model() to.
std::shared_ptr<Model> build_restored_model(const bool u8_zero_point = false) {
    const auto ids = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    const auto codes = std::make_shared<op::v0::Constant>(element::u2,
                                                          Shape{VOCAB, GROUPS, PACKED_GROUP_SIZE * 4},
                                                          packed_bytes(PACKED_GROUP_SIZE).data());
    const auto codes_f16 = std::make_shared<op::v0::Convert>(codes, element::f16);
    const auto centered = std::make_shared<op::v1::Subtract>(codes_f16, make_zero_point(1.f, u8_zero_point));
    const auto dequantized = std::make_shared<op::v1::Multiply>(centered, make_scales());
    const auto table = std::make_shared<op::v1::Reshape>(
        dequantized,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{VOCAB, HIDDEN}),
        false);
    const auto table_f32 = std::make_shared<op::v0::Convert>(table, element::f32);
    const auto rows =
        std::make_shared<op::v8::Gather>(table_f32, ids, op::v0::Constant::create(element::i32, Shape{}, {0}));
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(rows)}, ParameterVector{ids});
}

Tensor evaluate(const std::shared_ptr<Model>& model, const std::vector<int32_t>& ids) {
    Tensor ids_tensor(element::i32, Shape{2, 3});
    std::copy(ids.begin(), ids.end(), ids_tensor.data<int32_t>());
    TensorVector outputs{Tensor()};
    EXPECT_TRUE(model->evaluate(outputs, TensorVector{ids_tensor}));
    return outputs.front();
}

}  // namespace

TEST_F(TransformationTestsF, RestoreU2GatherDequant_RewritesPackedGatherToGatherOverDequantizedU2Vocabulary) {
    model = build_packed_gather_model();
    manager.register_pass<npuw::RestoreU2GatherDequant>();
    model_ref = build_restored_model();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_RestoresZeroPointConvertedFromU8) {
    PackedModelParams params;
    params.u8_zero_point = true;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
    model_ref = build_restored_model(true);
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphWithUnexpectedZeroPoint) {
    PackedModelParams params;
    params.zero_point = 4.f;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphWithNegativeZeroPoint) {
    PackedModelParams params;
    params.zero_point = -1.f;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphWithFractionalZeroPoint) {
    PackedModelParams params;
    params.zero_point = 0.5f;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphWithUnexpectedBitOrder) {
    PackedModelParams params;
    params.shift_step = 1;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphWithCodeMajorUnpacking) {
    PackedModelParams params;
    params.code_major = true;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST_F(TransformationTestsF, RestoreU2GatherDequant_KeepsGraphOfVocabSmallerThanHostGatherThreshold) {
    PackedModelParams params;
    params.packed_group_size = SMALL_PACKED_GROUP_SIZE;
    model = build_packed_gather_model(params);
    manager.register_pass<npuw::RestoreU2GatherDequant>();
}

TEST(RestoreU2GatherDequantPassTest, DequantizesLowBitsFirst) {
    const auto model = build_packed_gather_model();
    npuw::RestoreU2GatherDequant().run_on_model(model);
    model->validate_nodes_and_infer_types();

    const std::vector<int32_t> ids{0, 3, 4, 2, 2, 1};
    const auto actual = evaluate(model, ids);

    ASSERT_EQ(actual.get_shape(), (Shape{2, 3, HIDDEN}));
    const auto* actual_data = actual.data<float>();
    for (size_t row = 0; row < ids.size(); ++row) {
        for (size_t column = 0; column < HIDDEN; ++column) {
            const size_t group = column / (PACKED_GROUP_SIZE * 4);
            const size_t position = column % (PACKED_GROUP_SIZE * 4);
            const size_t group_index = static_cast<size_t>(ids[row]) * GROUPS + group;
            const int code = (packed_byte(group_index * PACKED_GROUP_SIZE + position / 4) >> (2 * (position % 4))) & 3;
            const float expected = static_cast<float>(code - 1) * scale_value(group_index);
            EXPECT_EQ(actual_data[row * HIDDEN + column], expected) << "row " << row << " column " << column;
        }
    }
}
