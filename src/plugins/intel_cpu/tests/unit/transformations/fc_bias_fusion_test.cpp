// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/fc_bias_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/fully_connected.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {

std::shared_ptr<ov::Node> make_fc(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& bias) {
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, {1});
    return std::make_shared<ov::op::internal::FullyConnected>(input, weights, bias);
}

std::shared_ptr<ov::Node> no_bias() {
    return std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
}

std::shared_ptr<ov::Node> bias_const(const std::vector<float>& values) {
    return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{values.size()}, values);
}

}  // namespace

class FullyConnectedBiasFusionTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<FullyConnectedBiasFusion>();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
};

TEST_F(FullyConnectedBiasFusionTests, SingleAdd) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
        auto add = std::make_shared<ov::op::v1::Add>(make_fc(input, no_bias()), bias_const({1, 2, 3, 4}));
        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
        auto fc = make_fc(input, bias_const({1, 2, 3, 4}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
}

// The second Add must be added to the bias fused from the first one, not replace it.
TEST_F(FullyConnectedBiasFusionTests, TwoAddsAccumulateBias) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
        auto add1 = std::make_shared<ov::op::v1::Add>(make_fc(input, no_bias()), bias_const({1, 2, 3, 4}));
        auto add2 = std::make_shared<ov::op::v1::Add>(add1, bias_const({10, 20, 30, 40}));
        model = std::make_shared<ov::Model>(ov::OutputVector{add2}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
        auto fc = make_fc(input, bias_const({11, 22, 33, 44}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
}

// An FC that already has a bias of a different shape keeps it and leaves the Add in place.
TEST_F(FullyConnectedBiasFusionTests, ExistingBiasShapeMismatch) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
        auto fc_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {1, 2, 3, 4});
        auto add = std::make_shared<ov::op::v1::Add>(make_fc(input, fc_bias), bias_const({10, 20, 30, 40}));
        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
}
