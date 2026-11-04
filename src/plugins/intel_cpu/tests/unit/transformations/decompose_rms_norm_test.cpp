// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/decompose_rms_norm.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "ov_ops/rms.hpp"

using namespace ov;
using namespace ov::op;

namespace {
constexpr double kEpsilon = 1e-5;

std::shared_ptr<Node> make_decomposed_rms(const Output<Node>& data, const Output<Node>& gamma) {
    const auto prc = data.get_element_type();
    auto power = std::make_shared<v1::Power>(data, v0::Constant::create(prc, {}, std::vector<float>{2.F}));
    auto mean = std::make_shared<v1::ReduceMean>(power, v0::Constant::create(element::i32, Shape{1}, {-1}), true);
    auto add_eps = std::make_shared<v1::Add>(mean, v0::Constant::create(prc, {}, {kEpsilon}));
    auto inv = std::make_shared<v1::Power>(std::make_shared<v0::Sqrt>(add_eps), v0::Constant::create(prc, {}, {-1}));
    return std::make_shared<v1::Multiply>(gamma, std::make_shared<v1::Multiply>(data, inv));
}
}  // namespace

// f16 data with f32 gamma: RMS outputs f32, so the decomposition must run in f32
TEST_F(TransformationTestsF, DecomposeRMSNormMixedPrecision) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto data = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, 8});
        auto gamma = v0::Constant::create(element::f32, Shape{8}, {1.5F});
        auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, kEpsilon, element::f32);
        model = std::make_shared<Model>(OutputVector{rms}, ParameterVector{data});
        manager.register_pass<ov::intel_cpu::DecomposeRMSNorm>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, -1, 8});
        auto gamma = v0::Constant::create(element::f32, Shape{8}, {1.5F});
        auto norm = make_decomposed_rms(std::make_shared<v0::Convert>(data, element::f32), gamma);
        model_ref = std::make_shared<Model>(OutputVector{norm}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, DecomposeRMSNormSamePrecision) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 8});
        auto gamma = v0::Constant::create(element::f32, Shape{8}, {1.5F});
        auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, kEpsilon, element::f32);
        model = std::make_shared<Model>(OutputVector{rms}, ParameterVector{data});
        manager.register_pass<ov::intel_cpu::DecomposeRMSNorm>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 8});
        auto gamma = v0::Constant::create(element::f32, Shape{8}, {1.5F});
        model_ref = std::make_shared<Model>(OutputVector{make_decomposed_rms(data, gamma)}, ParameterVector{data});
    }
}
