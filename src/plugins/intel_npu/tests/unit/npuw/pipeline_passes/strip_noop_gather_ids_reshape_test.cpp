// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/strip_noop_gather_ids_reshape.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"

namespace {

using namespace ov;

constexpr size_t kVocab = 4;
constexpr size_t kHidden = 2048;
constexpr size_t kSmallHidden = 8;

// ids -> Convert(i32) -> Reshape[1,1,1,N] -> Squeeze{0,1} -> Gather(Constant vocab[V,H])
std::shared_ptr<Model> make_model(const size_t hidden) {
    std::vector<float> vocab_values(kVocab * hidden);
    for (size_t i = 0; i < vocab_values.size(); ++i) {
        vocab_values[i] = static_cast<float>(i % 17);
    }
    const auto vocab = op::v0::Constant::create(element::f32, Shape{kVocab, hidden}, vocab_values);

    const auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 3});
    auto indices = std::make_shared<op::v0::Squeeze>(
        std::make_shared<op::v1::Reshape>(
            std::make_shared<op::v0::Convert>(ids, element::i32),
            op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, 1, 3}),
            false),
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, 1}));
    auto gather = std::make_shared<op::v8::Gather>(vocab, indices, op::v0::Constant::create(element::i32, {}, {0}));
    return std::make_shared<Model>(OutputVector{gather}, ParameterVector{ids});
}

std::vector<float> evaluate(const std::shared_ptr<Model>& model) {
    Tensor ids(element::i64, Shape{1, 3});
    ids.data<int64_t>()[0] = 2;
    ids.data<int64_t>()[1] = 0;
    ids.data<int64_t>()[2] = 3;
    TensorVector outputs{Tensor(element::f32, Shape{})};
    EXPECT_TRUE(model->evaluate(outputs, TensorVector{ids}));
    const auto* data = outputs.front().data<float>();
    return std::vector<float>(data, data + outputs.front().get_size());
}

}  // namespace

TEST(StripNoopGatherIdsReshapeTest, FeedsGatherStraightFromConvertedIds) {
    const auto reference = make_model(kHidden);
    const auto model = reference->clone();
    ASSERT_TRUE(ov::npuw::StripNoopGatherIdsReshape().run_on_model(model));
    model->validate_nodes_and_infer_types();

    const auto ops = model->get_ordered_ops();
    const auto gather = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& op) {
        return ov::is_type<op::v8::Gather>(op);
    });
    ASSERT_NE(gather, ops.end());
    EXPECT_TRUE(ov::is_type<op::v0::Convert>((*gather)->input_value(1).get_node()));
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const std::shared_ptr<Node>& op) {
                                return ov::is_type<op::v0::Squeeze>(op);
                            }),
              0);

    const auto expected = evaluate(reference);
    const auto actual = evaluate(model);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

TEST(StripNoopGatherIdsReshapeTest, KeepsIdsReshapeOfVocabSmallerThanHostGatherThreshold) {
    const auto model = make_model(kSmallHidden);
    EXPECT_FALSE(ov::npuw::StripNoopGatherIdsReshape().run_on_model(model));
}
