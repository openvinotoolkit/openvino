// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "ov_ops/rms.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr size_t hidden = 64;
constexpr float eps = 1e-6f;

// x / sqrt(ReduceSum(x^2, axis) + eps) * gamma, as an ONNX LpNormalization(p=2) followed by a scale exports it.
std::shared_ptr<ov::Node> l2_norm(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, int64_t axis = -1, bool keep_dims = true) {
    auto power = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(x.get_element_type(), {}, {2.f}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, axes, keep_dims);
    auto add = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(x.get_element_type(), {}, {eps}));
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add);
    auto div = std::make_shared<ov::op::v1::Divide>(x, sqrt);
    return std::make_shared<ov::op::v1::Multiply>(div, gamma);
}

std::shared_ptr<ov::Node> rms(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, ov::element::Type out_type) {
    auto inv_sqrt_n = ov::op::v0::Constant::create(gamma.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(hidden))});
    auto scaled_gamma = std::make_shared<ov::op::v1::Multiply>(gamma, inv_sqrt_n);
    return std::make_shared<ov::op::internal::RMS>(x, scaled_gamma, eps / static_cast<float>(hidden), out_type);
}

}  // namespace

TEST_F(TransformationTestsF, L2NormToRMS) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto norm = l2_norm(x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// f16 activations normalized in f32 and converted back: RMS takes the f16 input and writes f16 directly.
TEST_F(TransformationTestsF, L2NormToRMSWithConverts) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto x_f32 = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto norm = l2_norm(x_f32, gamma);
        auto out = std::make_shared<ov::op::v0::Convert>(norm, ov::element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f16)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Not normalized over the last axis: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNotLastAxis) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    auto norm = l2_norm(x, gamma, 1);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// Without keep_dims the reduced tensor does not broadcast against x the way RMS needs: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNoKeepDims) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma, -1, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// A row too long for one work group of the RMS kernel (hidden / 8 > max work group size): left alone.
TEST_F(TransformationTestsF, L2NormToRMSRowTooLong) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(4);
}
