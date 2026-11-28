// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/insert_vocab_sub128.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {

using namespace ov;

constexpr size_t kVocab = 4;
constexpr size_t kHidden = 2;

// A u8 asymmetric [V, H] vocabulary with a per-row zero point and scale.
npuw::Vocab make_vocab(uint8_t weight_value = 200) {
    return {op::v0::Constant::create(element::u8,
                                     Shape{kVocab, kHidden},
                                     std::vector<uint8_t>(kVocab * kHidden, weight_value)),
            op::v0::Constant::create(element::u8, Shape{kVocab, 1}, std::vector<uint8_t>(kVocab, 128)),
            op::v0::Constant::create(element::f16, Shape{kVocab, 1}, std::vector<float>(kVocab, 0.5f))};
}

// A second set of Constant nodes over the buffers of `vocab`, as for a tied vocabulary read from IR.
npuw::Vocab view_of(const npuw::Vocab& vocab) {
    const auto view = [](const std::shared_ptr<op::v0::Constant>& constant) {
        return std::make_shared<op::v0::Constant>(
            Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr())));
    };
    return {view(vocab.weight), view(vocab.zerop), view(vocab.scale)};
}

// (Convert(W) [- 128] - Convert(Z) [- 128]) * S [-> Convert]. The shifted form is what the passes produce.
Output<Node> dequantize(const npuw::Vocab& vocab, bool shifted, bool convert_output) {
    const auto weight_convert = std::make_shared<op::v0::Convert>(vocab.weight, element::f16);
    const auto zerop_convert = std::make_shared<op::v0::Convert>(vocab.zerop, element::f16);
    Output<Node> weight = weight_convert;
    Output<Node> zerop = zerop_convert;
    if (shifted) {
        mark_as_decompression(weight_convert);
        mark_as_decompression(zerop_convert);
        const auto shift = op::v0::Constant::create(element::f16, Shape{}, {128});
        const auto shifted_weight = std::make_shared<op::v1::Subtract>(weight_convert, shift);
        const auto shifted_zerop = std::make_shared<op::v1::Subtract>(zerop_convert, shift);
        shifted_weight->get_rt_info()[npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        shifted_zerop->get_rt_info()[npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        weight = shifted_weight;
        zerop = shifted_zerop;
    }
    const auto subtract = std::make_shared<op::v1::Subtract>(weight, zerop);
    const auto scaled = std::make_shared<op::v1::Multiply>(subtract, vocab.scale);
    if (!convert_output) {
        return scaled;
    }
    return std::make_shared<op::v0::Convert>(scaled, element::f32);
}

std::shared_ptr<Model> make_embedding_model(const npuw::Vocab& vocab, bool shifted, bool convert_output = true) {
    const auto ids = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 1});
    const auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto gather = std::make_shared<op::v8::Gather>(dequantize(vocab, shifted, convert_output), ids, axis);
    return std::make_shared<Model>(OutputVector{gather}, ParameterVector{ids});
}

std::shared_ptr<Model> make_lm_head_model(const npuw::Vocab& vocab, bool shifted, bool convert_output = true) {
    const auto hidden =
        std::make_shared<op::v0::Parameter>(convert_output ? element::f32 : element::f16, Shape{1, 1, kHidden});
    const auto matmul =
        std::make_shared<op::v0::MatMul>(hidden, dequantize(vocab, shifted, convert_output), false, true);
    return std::make_shared<Model>(OutputVector{matmul}, ParameterVector{hidden});
}

class InsertEmbeddingSub128Test : public TransformationTestsF, public ::testing::WithParamInterface<bool> {
public:
    InsertEmbeddingSub128Test() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    static std::string getTestCaseName(const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "WithConvert" : "WithoutConvert";
    }
};

TEST_P(InsertEmbeddingSub128Test, ShiftsGatherOverLmHeadVocabulary) {
    const auto vocab = make_vocab();
    model = make_embedding_model(vocab, false, GetParam());
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{vocab});
    model_ref = make_embedding_model(vocab, true, GetParam());
}

// A tied vocabulary read from IR: the embedding and the LM head have different Constant nodes over one buffer.
TEST_P(InsertEmbeddingSub128Test, ShiftsGatherOverDistinctConstantsOfOneBuffer) {
    const auto vocab = make_vocab();
    model = make_embedding_model(view_of(vocab), false, GetParam());
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{vocab});
    model_ref = make_embedding_model(vocab, true, GetParam());
}

// The shifted Gather does not match again.
TEST_P(InsertEmbeddingSub128Test, RunningTwiceShiftsOnce) {
    const auto vocab = make_vocab();
    model = make_embedding_model(vocab, false, GetParam());
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{vocab});
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{vocab});
    model_ref = make_embedding_model(vocab, true, GetParam());
}

// A table with the same values but its own storage is not shared with the LM head.
TEST_P(InsertEmbeddingSub128Test, KeepsGatherOverSeparateVocabulary) {
    model = make_embedding_model(make_vocab(), false, GetParam());
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{make_vocab()});
}

TEST_P(InsertEmbeddingSub128Test, KeepsGatherWithoutLmHeadVocabulary) {
    model = make_embedding_model(make_vocab(), false, GetParam());
    manager.register_pass<npuw::InsertEmbeddingSub128>(std::vector<npuw::Vocab>{});
}

INSTANTIATE_TEST_SUITE_P(OutputConvert,
                         InsertEmbeddingSub128Test,
                         ::testing::Bool(),
                         InsertEmbeddingSub128Test::getTestCaseName);

class InsertVocabSub128GraphTest : public TransformationTestsF, public ::testing::WithParamInterface<bool> {
public:
    InsertVocabSub128GraphTest() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    static std::string getTestCaseName(const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "WithConvert" : "WithoutConvert";
    }
};

TEST_P(InsertVocabSub128GraphTest, ShiftsLmHeadVocabulary) {
    const auto vocab = make_vocab();
    model = make_lm_head_model(vocab, false, GetParam());
    manager.register_pass<npuw::InsertVocabSub128>();
    model_ref = make_lm_head_model(vocab, true, GetParam());
}

TEST_P(InsertVocabSub128GraphTest, RunningTwiceShiftsOnce) {
    const auto vocab = make_vocab();
    model = make_lm_head_model(vocab, false, GetParam());
    manager.register_pass<npuw::InsertVocabSub128>();
    manager.register_pass<npuw::InsertVocabSub128>();
    model_ref = make_lm_head_model(vocab, true, GetParam());
}

INSTANTIATE_TEST_SUITE_P(OutputConvert,
                         InsertVocabSub128GraphTest,
                         ::testing::Bool(),
                         InsertVocabSub128GraphTest::getTestCaseName);

bool same_vocab(const npuw::Vocab& lhs, const npuw::Vocab& rhs) {
    return lhs.weight == rhs.weight && lhs.zerop == rhs.zerop && lhs.scale == rhs.scale;
}

TEST(InsertVocabSub128ReportTest, ReportsShiftedVocabulary) {
    const auto vocab = make_vocab();
    std::vector<npuw::Vocab> shifted;

    EXPECT_TRUE(npuw::InsertVocabSub128(shifted).run_on_model(make_lm_head_model(vocab, false)));
    ASSERT_EQ(shifted.size(), 1u);
    EXPECT_TRUE(same_vocab(shifted.front(), vocab));
}

TEST(InsertVocabSub128ReportTest, ReportsNothingForAShiftedVocabulary) {
    std::vector<npuw::Vocab> shifted;

    EXPECT_FALSE(npuw::InsertVocabSub128(shifted).run_on_model(make_lm_head_model(make_vocab(), true)));
    EXPECT_TRUE(shifted.empty());
}

// The sequence of LLMCompiledModel: the LM head model is shifted first, then the embedding of the other
// models with the vocabulary that was reported. Both readers end up with the same shifted graph.
TEST(InsertVocabSub128ReportTest, EmbeddingAndLmHeadAreShiftedTogether) {
    const auto vocab = make_vocab();
    const auto lm_head = make_lm_head_model(vocab, false);
    const auto embedding = make_embedding_model(view_of(vocab), false);

    std::vector<npuw::Vocab> shifted;
    ASSERT_TRUE(npuw::InsertVocabSub128(shifted).run_on_model(lm_head));
    EXPECT_TRUE(npuw::InsertEmbeddingSub128(shifted).run_on_model(embedding));

    const auto comparator = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::RUNTIME_KEYS);
    const auto lm_head_result = comparator.compare(lm_head, make_lm_head_model(vocab, true));
    EXPECT_TRUE(lm_head_result.valid) << lm_head_result.message;
    const auto embedding_result = comparator.compare(embedding, make_embedding_model(vocab, true));
    EXPECT_TRUE(embedding_result.valid) << embedding_result.message;
}

}  // namespace
