// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/decompose_rms_norm.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
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

using DecomposeRMSNormParams = std::tuple<element::Type,  // data precision, the RMS output is f32
                                          bool>;          // with gamma

std::shared_ptr<Node> make_decomposed_rms(const Output<Node>& data, const std::shared_ptr<Node>& gamma) {
    const auto prc = data.get_element_type();
    auto power = std::make_shared<v1::Power>(data, v0::Constant::create(prc, {}, std::vector<float>{2.F}));
    auto mean = std::make_shared<v1::ReduceMean>(power, v0::Constant::create(element::i32, Shape{1}, {-1}), true);
    auto add_eps = std::make_shared<v1::Add>(mean, v0::Constant::create(prc, {}, {kEpsilon}));
    auto inv = std::make_shared<v1::Power>(std::make_shared<v0::Sqrt>(add_eps), v0::Constant::create(prc, {}, {-1}));
    std::shared_ptr<Node> norm = std::make_shared<v1::Multiply>(data, inv);
    return gamma ? std::make_shared<v1::Multiply>(gamma, norm) : norm;
}

class DecomposeRMSNormTest : public TransformationTestsF, public testing::WithParamInterface<DecomposeRMSNormParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DecomposeRMSNormParams>& obj) {
        const auto& [data_prc, with_gamma] = obj.param;
        return "data=" + data_prc.get_type_name() + (with_gamma ? "_gamma" : "_no_gamma");
    }
};

// RMS outputs f32 for f16 data too, so the decomposition must run in the output type
TEST_P(DecomposeRMSNormTest, ComputesInOutputPrecision) {
    const auto& [data_prc, with_gamma] = GetParam();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto data = std::make_shared<v0::Parameter>(data_prc, PartialShape{-1, -1, 8});
        auto rms = with_gamma
                       ? std::make_shared<ov::op::internal::RMS>(data,
                                                                 v0::Constant::create(element::f32, Shape{8}, {1.5F}),
                                                                 kEpsilon,
                                                                 element::f32)
                       : std::make_shared<ov::op::internal::RMS>(data, kEpsilon, element::f32);
        model = std::make_shared<Model>(OutputVector{rms}, ParameterVector{data});
        manager.register_pass<ov::intel_cpu::DecomposeRMSNorm>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(data_prc, PartialShape{-1, -1, 8});
        Output<Node> input = data;
        if (data_prc != element::f32) {
            input = std::make_shared<v0::Convert>(data, element::f32);
        }
        auto gamma = with_gamma ? v0::Constant::create(element::f32, Shape{8}, {1.5F}) : nullptr;
        model_ref = std::make_shared<Model>(OutputVector{make_decomposed_rms(input, gamma)}, ParameterVector{data});
    }
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DecomposeRMSNormTest,
                         testing::Combine(testing::Values(element::f16, element::f32), testing::Bool()),
                         DecomposeRMSNormTest::getTestCaseName);
}  // namespace
