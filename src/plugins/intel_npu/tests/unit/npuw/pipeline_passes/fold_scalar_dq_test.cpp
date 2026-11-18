// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/fold_scalar_dq.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"

namespace {

using namespace ov;

constexpr uint16_t kWeight = 3;
constexpr uint16_t kZeroPoint = 65535;
constexpr float kScale = 0.25f;

struct DQParams {
    bool with_subtract = true;
    bool swap_subtract = false;
    bool swap_multiply = false;
    bool scale_convert = false;
    bool weight_parameter = false;
    bool weight_convert = true;
    Shape weight_shape{1};
};

float expected_value(const DQParams& p) {
    const float w = kWeight, zp = kZeroPoint;
    const float shifted = !p.with_subtract ? w : p.swap_subtract ? zp - w : w - zp;
    return shifted * kScale;
}

// x * DQ(weight), where DQ = Convert(weight) -> [Subtract(Convert(zp))] -> Multiply(scale)
std::shared_ptr<Model> build_model(const DQParams& p, bool folded) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});
    ParameterVector params{x};

    std::shared_ptr<Node> dq;
    if (folded) {
        dq = op::v0::Constant::create(element::f32, p.weight_shape, {expected_value(p)});
    } else {
        std::shared_ptr<Node> weight;
        if (p.weight_parameter) {
            auto weight_param = std::make_shared<op::v0::Parameter>(element::u16, p.weight_shape);
            params.push_back(weight_param);
            weight = weight_param;
        } else {
            weight =
                op::v0::Constant::create(p.weight_convert ? element::u16 : element::f32, p.weight_shape, {kWeight});
        }
        std::shared_ptr<Node> value =
            p.weight_convert ? std::make_shared<op::v0::Convert>(weight, element::f32) : weight;
        if (p.with_subtract) {
            auto zp = std::make_shared<op::v0::Convert>(op::v0::Constant::create(element::u16, Shape{}, {kZeroPoint}),
                                                        element::f32);
            value = p.swap_subtract ? std::make_shared<op::v1::Subtract>(zp, value)
                                    : std::make_shared<op::v1::Subtract>(value, zp);
        }
        std::shared_ptr<Node> scale =
            op::v0::Constant::create(p.scale_convert ? element::f16 : element::f32, Shape{}, {kScale});
        if (p.scale_convert) {
            scale = std::make_shared<op::v0::Convert>(scale, element::f32);
        }
        dq = p.swap_multiply ? std::make_shared<op::v1::Multiply>(scale, value)
                             : std::make_shared<op::v1::Multiply>(value, scale);
    }

    auto out = std::make_shared<op::v1::Multiply>(x, dq);
    return std::make_shared<Model>(OutputVector{out}, params);
}

class FoldScalarDQTest : public TransformationTestsF,
                         public ::testing::WithParamInterface<std::tuple<bool, bool, bool, bool>> {
public:
    FoldScalarDQTest() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
};

TEST_P(FoldScalarDQTest, FoldsChain) {
    DQParams p;
    std::tie(p.with_subtract, p.swap_subtract, p.swap_multiply, p.scale_convert) = GetParam();
    model = build_model(p, false);
    manager.register_pass<npuw::FoldScalarDQ>();
    model_ref = build_model(p, true);
}

INSTANTIATE_TEST_SUITE_P(NPUW,
                         FoldScalarDQTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool(), ::testing::Bool()),
                         [](const ::testing::TestParamInfo<std::tuple<bool, bool, bool, bool>>& info) {
                             return std::string(std::get<0>(info.param) ? "WithZP" : "NoZP") +
                                    (std::get<1>(info.param) ? "_SwappedSub" : "") +
                                    (std::get<2>(info.param) ? "_SwappedMul" : "") +
                                    (std::get<3>(info.param) ? "_ScaleConvert" : "");
                         });

TEST_F(TransformationTestsF, FoldScalarDQ_NotAppliedForParameterWeight) {
    DQParams p;
    p.weight_parameter = true;
    model = build_model(p, false);
    manager.register_pass<npuw::FoldScalarDQ>();
}

TEST_F(TransformationTestsF, FoldScalarDQ_NotAppliedForMultiElementWeight) {
    DQParams p;
    p.weight_shape = Shape{4};
    model = build_model(p, false);
    manager.register_pass<npuw::FoldScalarDQ>();
}

TEST_F(TransformationTestsF, FoldScalarDQ_NotAppliedWithoutWeightConvert) {
    DQParams p;
    p.weight_convert = false;
    model = build_model(p, false);
    manager.register_pass<npuw::FoldScalarDQ>();
}

}  // namespace
