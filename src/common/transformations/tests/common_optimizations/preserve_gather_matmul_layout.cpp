// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/preserve_gather_matmul_layout.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace {

std::shared_ptr<ov::Model> make_model(const ov::PartialShape& activation_shape, bool reshape) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, activation_shape);
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 6, 8});
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 2});
    ov::Output<ov::Node> input = activation;
    if (reshape) {
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, -1, 0});
        input = std::make_shared<ov::op::v1::Reshape>(input, pattern, true);
    }
    auto bias = ov::op::v0::Constant::create(ov::element::f32, {0}, std::vector<float>{});
    auto gmm = std::make_shared<ov::op::internal::GatherMatmul>(input, weights, indices, bias);
    return std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{activation, weights, indices});
}

class PreserveGatherMatmulLayoutTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::PreserveGatherMatmulLayout>();
        manager.register_pass<ov::pass::PreserveGatherMatmulLayout>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_F(PreserveGatherMatmulLayoutTest, PreservesSharedExpertActivation) {
    model = make_model({1, -1, 8}, false);
    model_ref = make_model({1, -1, 8}, true);
}

TEST_F(PreserveGatherMatmulLayoutTest, LeavesExpertSpecificActivation) {
    model = make_model({2, -1, 8}, false);
}

TEST_F(PreserveGatherMatmulLayoutTest, LeavesUnknownExpertAxis) {
    model = make_model({-1, -1, 8}, false);
}

}  // namespace
