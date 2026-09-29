// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/cpu_opset/common/pass/decompose_rms_norm.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {
std::shared_ptr<ov::Node> makeDecomposedRMS(const ov::Output<ov::Node>& data,
                                            const ov::Output<ov::Node>& gamma,
                                            double eps,
                                            const ov::element::Type& data_precision) {
    auto power_const = ov::op::v0::Constant::create(data_precision, {}, {2.F});
    auto power = std::make_shared<ov::op::v1::Power>(data, power_const);
    auto mean_axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);
    auto eps_const = ov::op::v0::Constant::create(data_precision, {}, {eps});
    auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps_const);
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    auto div_const = ov::op::v0::Constant::create(data_precision, {}, {-1});
    auto div = std::make_shared<ov::op::v1::Power>(sqrt, div_const);
    std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v1::Multiply>(data, div);
    auto aligned_gamma = gamma;
    if (gamma.get_element_type() != data_precision) {
        aligned_gamma = std::make_shared<ov::op::v0::Convert>(gamma, data_precision);
    }
    return std::make_shared<ov::op::v1::Multiply>(aligned_gamma, result);
}
}  // namespace

class DecomposeRMSNormTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DecomposeRMSNorm>();
    }
};

TEST_F(DecomposeRMSNormTests, SamePrecisionDataAndGamma) {
    const auto precision = ov::element::f32;
    const ov::PartialShape shape{-1, -1, 16};
    {
        auto data = std::make_shared<ov::opset1::Parameter>(precision, shape);
        auto gamma = ov::op::v0::Constant::create(precision, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, 1e-5, precision);
        model = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::opset1::Parameter>(precision, shape);
        auto gamma = ov::op::v0::Constant::create(precision, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto result = makeDecomposedRMS(data, gamma, 1e-5, precision);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data});
    }
    disable_rt_info_check();
    disable_result_friendly_names_check();
}

TEST_F(DecomposeRMSNormTests, MixedPrecisionDataF32GammaBf16) {
    // Gemma-style RMSNorm: hidden states upcast to f32 for the norm math,
    // but the learned gamma constant stays bf16. Without an explicit
    // alignment Convert, the final Multiply(bf16, f32) is invalid and
    // OpenVINO's core validator rejects the graph outright.
    const ov::PartialShape shape{-1, -1, 16};
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto gamma = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, 1e-5, ov::element::f32);
        model = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto gamma = ov::op::v0::Constant::create(ov::element::bf16, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto result = makeDecomposedRMS(data, gamma, 1e-5, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data});
    }
    disable_rt_info_check();
    disable_result_friendly_names_check();
}

TEST_F(DecomposeRMSNormTests, OutputTypeNarrowerThanDataPrecision) {
    // The fused RMS op can request a narrower output_type than the f32
    // data path computes in (e.g. surrounding graph runs in bf16); the
    // decomposition must cast the final result down to match.
    const ov::PartialShape shape{-1, -1, 16};
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, 1e-5, ov::element::bf16);
        model = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16}, std::vector<float>(16, 1.f));
        auto result = makeDecomposedRMS(data, gamma, 1e-5, ov::element::f32);
        auto converted = std::make_shared<ov::op::v0::Convert>(result, ov::element::bf16);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{converted}, ov::ParameterVector{data});
    }
    disable_rt_info_check();
    disable_result_friendly_names_check();
}
