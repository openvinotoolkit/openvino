// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/restore_u2_gather_dequant.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

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

constexpr size_t VOCAB = 5;
constexpr size_t GROUPS = 2;
constexpr size_t PACKED_GROUP_SIZE = 2;  // bytes, four 2-bit codes each
constexpr size_t HIDDEN = GROUPS * PACKED_GROUP_SIZE * 4;

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        count += ov::is_type<T>(op) ? 1u : 0u;
    }
    return count;
}

uint8_t packed_byte(size_t index) {
    return static_cast<uint8_t>((index * 37 + 11) & 0xFF);
}

float scale_value(size_t index) {
    return 0.25f * static_cast<float>(index + 1);
}

// Mirrors the packed gather of the GGUF frontend: Gather over u8 rows, then the bitwise unpacking of
// the 2-bit codes, the zero point, and Gather over the scales.
// `code_major` shifts along the axis before the packed bytes, which orders the codes differently from a u2 Constant.
// `u8_zero_point` builds the zero point as Convert(Constant u8), the form the GGUF frontend emits.
std::shared_ptr<ov::Model> build_packed_gather_model(const float zero_point = 1.f,
                                                     const int32_t shift_step = 2,
                                                     const bool code_major = false,
                                                     const bool u8_zero_point = false) {
    std::vector<uint8_t> bytes(VOCAB * GROUPS * PACKED_GROUP_SIZE);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = packed_byte(i);
    }
    std::vector<ov::float16> scales(VOCAB * GROUPS);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = ov::float16(scale_value(i));
    }
    const auto packed =
        ov::op::v0::Constant::create(ov::element::u8, ov::Shape{VOCAB, GROUPS, PACKED_GROUP_SIZE}, bytes);
    const auto scales_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{VOCAB, GROUPS, 1}, scales);
    const auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, -1});
    const auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    const auto selected = std::make_shared<ov::op::v8::Gather>(packed, ids, axis);
    const auto widened = std::make_shared<ov::op::v0::Convert>(selected, ov::element::i32);
    const auto expanded = std::make_shared<ov::op::v1::Reshape>(
        widened,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{5},
                                     code_major ? std::vector<int64_t>{0, 0, GROUPS, 1, PACKED_GROUP_SIZE}
                                                : std::vector<int64_t>{0, 0, GROUPS, PACKED_GROUP_SIZE, 1}),
        true);
    const auto shifted = std::make_shared<ov::op::v15::BitwiseRightShift>(
        expanded,
        ov::op::v0::Constant::create(ov::element::i32,
                                     code_major ? ov::Shape{4, 1} : ov::Shape{4},
                                     std::vector<int32_t>{0, shift_step, 2 * shift_step, 3 * shift_step}));
    const auto codes =
        std::make_shared<ov::op::v13::BitwiseAnd>(shifted,
                                                  ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3}));
    const auto grouped = std::make_shared<ov::op::v1::Reshape>(
        codes,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{4},
                                     std::vector<int64_t>{0, 0, GROUPS, PACKED_GROUP_SIZE * 4}),
        true);
    const auto values = std::make_shared<ov::op::v0::Convert>(grouped, ov::element::f16);
    std::shared_ptr<ov::Node> zero_point_node =
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {zero_point});
    if (u8_zero_point) {
        zero_point_node = std::make_shared<ov::op::v0::Convert>(
            ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {static_cast<uint8_t>(zero_point)}),
            ov::element::f16);
    }
    const auto centered = std::make_shared<ov::op::v1::Subtract>(values, zero_point_node);
    const auto selected_scales = std::make_shared<ov::op::v8::Gather>(scales_const, ids, axis);
    const auto dequantized = std::make_shared<ov::op::v1::Multiply>(centered, selected_scales);
    const auto restored = std::make_shared<ov::op::v1::Reshape>(
        dequantized,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, HIDDEN}),
        true);
    const auto embeds = std::make_shared<ov::op::v0::Convert>(restored, ov::element::f32);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(embeds)},
                                       ov::ParameterVector{ids});
}

ov::Tensor evaluate(const std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& ids) {
    ov::Tensor ids_tensor(ov::element::i32, ov::Shape{2, 3});
    std::copy(ids.begin(), ids.end(), ids_tensor.data<int32_t>());
    ov::TensorVector outputs{ov::Tensor()};
    EXPECT_TRUE(model->evaluate(outputs, ov::TensorVector{ids_tensor}));
    return outputs.front();
}

}  // namespace

TEST(RestoreU2GatherDequantPassTest, RewritesPackedGatherToGatherOverDequantizedU2Vocabulary) {
    const auto model = build_packed_gather_model();

    ov::npuw::RestoreU2GatherDequant().run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v15::BitwiseRightShift>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v13::BitwiseAnd>(model), 0u);
    ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_partial_shape(), (ov::PartialShape{-1, -1, HIDDEN}));

    const auto gather = model->get_results().front()->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(ov::is_type<ov::op::v8::Gather>(gather));
    const auto table = gather->input_value(0);
    EXPECT_EQ(table.get_shape(), (ov::Shape{VOCAB, HIDDEN}));
    EXPECT_EQ(table.get_element_type(), ov::element::f32);

    bool found_u2_weights = false;
    bool found_u2_zero_point = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
            if (constant->get_element_type() == ov::element::u2 &&
                constant->get_shape() == ov::Shape{VOCAB, GROUPS, PACKED_GROUP_SIZE * 4}) {
                found_u2_weights = true;
            }
            if (constant->get_element_type() == ov::element::u2 &&
                constant->get_shape() == ov::Shape{VOCAB, GROUPS, 1}) {
                found_u2_zero_point = true;
            }
        }
    }
    EXPECT_TRUE(found_u2_weights);
    EXPECT_TRUE(found_u2_zero_point);
}

TEST(RestoreU2GatherDequantPassTest, DequantizesLowBitsFirst) {
    const auto model = build_packed_gather_model();
    ov::npuw::RestoreU2GatherDequant().run_on_model(model);
    model->validate_nodes_and_infer_types();

    const std::vector<int32_t> ids{0, 3, 4, 2, 2, 1};
    const auto actual = evaluate(model, ids);

    ASSERT_EQ(actual.get_shape(), (ov::Shape{2, 3, HIDDEN}));
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

TEST(RestoreU2GatherDequantPassTest, KeepsGraphWithUnexpectedZeroPoint) {
    for (const float zero_point : {4.f, -1.f, 0.5f}) {
        const auto model = build_packed_gather_model(zero_point);

        ov::npuw::RestoreU2GatherDequant().run_on_model(model);
        model->validate_nodes_and_infer_types();

        EXPECT_EQ(count_ops<ov::op::v15::BitwiseRightShift>(model), 1u);
    }
}

TEST(RestoreU2GatherDequantPassTest, KeepsGraphWithUnexpectedBitOrder) {
    const auto model = build_packed_gather_model(1.f, 1);

    ov::npuw::RestoreU2GatherDequant().run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v15::BitwiseRightShift>(model), 1u);
}

TEST(RestoreU2GatherDequantPassTest, RestoresZeroPointConvertedFromU8) {
    const auto model = build_packed_gather_model(1.f, 2, false, true);
    const auto reference = build_packed_gather_model();

    ov::npuw::RestoreU2GatherDequant().run_on_model(model);
    ov::npuw::RestoreU2GatherDequant().run_on_model(reference);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v15::BitwiseRightShift>(model), 0u);
    const std::vector<int32_t> ids{0, 3, 4, 2, 2, 1};
    const auto actual = evaluate(model, ids);
    const auto expected = evaluate(reference, ids);
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.get_byte_size()), 0);
}

TEST(RestoreU2GatherDequantPassTest, KeepsGraphWithCodeMajorUnpacking) {
    const auto model = build_packed_gather_model(1.f, 2, true);

    ov::npuw::RestoreU2GatherDequant().run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v15::BitwiseRightShift>(model), 1u);
}
