// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/untangle_weight_decompression.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"

namespace {

// Tied embeddings as the PA model has them: one decompressed table feeds the
// embedding Gather and the LM head MatMul, and the 1-D token ids reach the
// Gather through an Unsqueeze. The table comes in the given weight format.
enum class TiedFormat { AsymU8, GroupedI8, F16 };

std::shared_ptr<ov::Model> make_tied_embedding_model(TiedFormat format, bool tied = true) {
    constexpr std::size_t vocab = 32, hidden = 16, groups = 4;
    const auto table = [&](const ov::Shape& shape) -> ov::Output<ov::Node> {
        std::vector<float> scales(vocab * groups);
        for (std::size_t i = 0; i < scales.size(); ++i) {
            scales[i] = 0.01f * static_cast<float>(i % 7 + 1);
        }
        switch (format) {
        case TiedFormat::AsymU8: {
            std::vector<uint8_t> data(vocab * hidden), zero(vocab);
            for (std::size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<uint8_t>(i * 37 % 251);
            }
            for (std::size_t i = 0; i < zero.size(); ++i) {
                zero[i] = static_cast<uint8_t>(100 + i);
            }
            auto weights = ov::op::v0::Constant::create(ov::element::u8, shape, data);
            auto zero_point = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{vocab, 1}, zero);
            auto shifted = std::make_shared<ov::op::v1::Subtract>(
                std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
                std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f32));
            scales.resize(vocab);
            auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{vocab, 1}, scales);
            return std::make_shared<ov::op::v1::Multiply>(shifted, scale);
        }
        case TiedFormat::GroupedI8: {
            std::vector<int8_t> data(vocab * hidden);
            for (std::size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<int8_t>(static_cast<int>(i * 29 % 200) - 100);
            }
            auto weights =
                ov::op::v0::Constant::create(ov::element::i8, ov::Shape{vocab, groups, hidden / groups}, data);
            auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{vocab, groups, 1}, scales);
            auto scaled =
                std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
                                                       scale);
            return std::make_shared<ov::op::v1::Reshape>(
                scaled,
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{vocab, hidden}),
                false);
        }
        case TiedFormat::F16:
        default: {
            std::vector<float> data(vocab * hidden);
            for (std::size_t i = 0; i < data.size(); ++i) {
                data[i] = 0.125f * static_cast<float>(static_cast<int>(i % 23) - 11);
            }
            auto weights = ov::op::v0::Constant::create(ov::element::f16, shape, data);
            return std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
        }
        }
    };
    const auto embed_table = table(ov::Shape{vocab, hidden});
    const auto head_table = tied ? embed_table : table(ov::Shape{vocab, hidden});

    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, hidden});
    auto ids_2d = std::make_shared<ov::op::v0::Unsqueeze>(ids, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
    auto embed = std::make_shared<ov::op::v8::Gather>(embed_table,
                                                      ids_2d,
                                                      ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto lm_head = std::make_shared<ov::op::v0::MatMul>(states, head_table, false, true);
    return std::make_shared<ov::Model>(ov::OutputVector{embed, lm_head}, ov::ParameterVector{ids, states});
}

// The ops and Constants an input is computed from.
void collect_sources(const ov::Output<ov::Node>& output, std::set<ov::Node*>& ops, std::set<ov::Node*>& constants) {
    const auto node = output.get_node_shared_ptr();
    if (ov::op::util::is_constant(node)) {
        constants.insert(node.get());
        return;
    }
    ops.insert(node.get());
    for (const auto& input : node->input_values()) {
        collect_sources(input, ops, constants);
    }
}

class UntangleWeightDecompressionTest : public testing::TestWithParam<TiedFormat> {};

TEST_P(UntangleWeightDecompressionTest, EveryReaderGetsItsOwnDecompression) {
    const auto reference = make_tied_embedding_model(GetParam());
    const auto model = reference->clone();
    EXPECT_TRUE(ov::npuw::UntangleWeightDecompression().run_on_model(model));

    std::shared_ptr<ov::Node> lm_head, embed;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::MatMul>(node)) {
            lm_head = node;
        } else if (ov::is_type<ov::op::v8::Gather>(node)) {
            embed = node;
        }
    }
    ASSERT_NE(lm_head, nullptr);
    ASSERT_NE(embed, nullptr);

    // No decompression op is shared, and both still read the one weight.
    std::set<ov::Node*> head_ops, head_constants, embed_ops, embed_constants;
    collect_sources(lm_head->input_value(1), head_ops, head_constants);
    collect_sources(embed->input_value(0), embed_ops, embed_constants);
    for (const auto op : head_ops) {
        EXPECT_EQ(embed_ops.count(op), 0u) << op->get_friendly_name() << " is shared";
    }
    EXPECT_EQ(head_constants, embed_constants);

    // A second run finds nothing left to untangle.
    EXPECT_FALSE(ov::npuw::UntangleWeightDecompression().run_on_model(model));

    ov::TensorVector inputs = {ov::Tensor(ov::element::i64, {4}), ov::Tensor(ov::element::f32, {2, 16})};
    for (std::size_t i = 0; i < 4; ++i) {
        inputs[0].data<int64_t>()[i] = static_cast<int64_t>(i * 7 % 32);
    }
    for (std::size_t i = 0; i < 32; ++i) {
        inputs[1].data<float>()[i] = 0.1f * static_cast<float>(i % 11) - 0.5f;
    }
    ov::TensorVector expected = {ov::Tensor(ov::element::f32, {4, 1, 16}), ov::Tensor(ov::element::f32, {2, 32})};
    ov::TensorVector actual = {ov::Tensor(ov::element::f32, {4, 1, 16}), ov::Tensor(ov::element::f32, {2, 32})};
    ASSERT_TRUE(reference->evaluate(expected, inputs));
    ASSERT_TRUE(model->evaluate(actual, inputs));
    for (std::size_t o = 0; o < expected.size(); ++o) {
        for (std::size_t i = 0; i < expected[o].get_size(); ++i) {
            EXPECT_FLOAT_EQ(expected[o].data<float>()[i], actual[o].data<float>()[i]) << "output " << o << " at " << i;
        }
    }
}

TEST_P(UntangleWeightDecompressionTest, WeightsThatAreNotSharedAreLeftAlone) {
    const auto model = make_tied_embedding_model(GetParam(), /*tied=*/false);
    const auto ops = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::UntangleWeightDecompression().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops);
}

INSTANTIATE_TEST_SUITE_P(WeightFormats,
                         UntangleWeightDecompressionTest,
                         testing::Values(TiedFormat::AsymU8, TiedFormat::GroupedI8, TiedFormat::F16));

}  // namespace
