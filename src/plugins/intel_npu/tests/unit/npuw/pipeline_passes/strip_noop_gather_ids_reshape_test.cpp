// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/strip_noop_gather_ids_reshape.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/ops.hpp"

namespace {

using namespace ov;

constexpr size_t kVocab = 4;
constexpr size_t kHidden = 2048;
constexpr size_t kSmallHidden = 8;
constexpr int64_t kTokens = 3;

std::shared_ptr<Node> make_vocab(const size_t hidden) {
    return op::v0::Constant::create(element::f32, Shape{kVocab, hidden}, std::vector<float>(kVocab * hidden, 1.f));
}

// ids -> Convert(i32) -> [Reshape[1,1,1,N] -> Squeeze(axes)] -> Gather(Constant vocab[V,H])
std::shared_ptr<Model> make_model(const size_t hidden,
                                  const bool with_reshape_squeeze,
                                  const std::vector<int64_t>& squeeze_axes = {0, 1}) {
    const auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, static_cast<size_t>(kTokens)});
    std::shared_ptr<Node> indices = std::make_shared<op::v0::Convert>(ids, element::i32);
    if (with_reshape_squeeze) {
        indices = std::make_shared<op::v0::Squeeze>(
            std::make_shared<op::v1::Reshape>(
                indices,
                op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, 1, kTokens}),
                false),
            op::v0::Constant::create(element::i64, Shape{squeeze_axes.size()}, squeeze_axes));
    }
    auto gather =
        std::make_shared<op::v8::Gather>(make_vocab(hidden), indices, op::v0::Constant::create(element::i32, {}, {0}));
    return std::make_shared<Model>(OutputVector{gather}, ParameterVector{ids});
}

}  // namespace

TEST_F(TransformationTestsF, StripNoopGatherIdsReshape_FeedsGatherStraightFromConvertedIds) {
    model = make_model(kHidden, true);
    manager.register_pass<npuw::StripNoopGatherIdsReshape>();
    model_ref = make_model(kHidden, false);
}

TEST_F(TransformationTestsF, StripNoopGatherIdsReshape_KeepsIdsReshapeOfVocabSmallerThanHostGatherThreshold) {
    model = make_model(kSmallHidden, true);
    manager.register_pass<npuw::StripNoopGatherIdsReshape>();
}

TEST_F(TransformationTestsF, StripNoopGatherIdsReshape_KeepsIdsReshapeThatChangesTheShape) {
    model = make_model(kHidden, true, {0});
    manager.register_pass<npuw::StripNoopGatherIdsReshape>();
}
