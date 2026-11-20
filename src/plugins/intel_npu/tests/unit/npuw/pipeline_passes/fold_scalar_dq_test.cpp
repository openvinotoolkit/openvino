// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/fold_scalar_dq.hpp"

#include <gtest/gtest.h>

#include <limits>
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
    bool zp_convert = true;
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

// x * DQ(weight), where DQ = Convert(weight) -> [Subtract([Convert](zp))] -> Multiply(scale)
std::shared_ptr<Model> getModel(const DQParams& p) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});
    ParameterVector params{x};

    std::shared_ptr<Node> weight;
    if (p.weight_parameter) {
        auto weight_param = std::make_shared<op::v0::Parameter>(element::u16, p.weight_shape);
        params.push_back(weight_param);
        weight = weight_param;
    } else {
        weight = op::v0::Constant::create(p.weight_convert ? element::u16 : element::f32, p.weight_shape, {kWeight});
    }
    std::shared_ptr<Node> value = p.weight_convert ? std::make_shared<op::v0::Convert>(weight, element::f32) : weight;
    if (p.with_subtract) {
        std::shared_ptr<Node> zp =
            op::v0::Constant::create(p.zp_convert ? element::u16 : element::f32, Shape{}, {kZeroPoint});
        if (p.zp_convert) {
            zp = std::make_shared<op::v0::Convert>(zp, element::f32);
        }
        value = p.swap_subtract ? std::make_shared<op::v1::Subtract>(zp, value)
                                : std::make_shared<op::v1::Subtract>(value, zp);
    }
    std::shared_ptr<Node> scale =
        op::v0::Constant::create(p.scale_convert ? element::f16 : element::f32, Shape{}, {kScale});
    if (p.scale_convert) {
        scale = std::make_shared<op::v0::Convert>(scale, element::f32);
    }
    auto dq = p.swap_multiply ? std::make_shared<op::v1::Multiply>(scale, value)
                              : std::make_shared<op::v1::Multiply>(value, scale);

    auto out = std::make_shared<op::v1::Multiply>(x, dq);
    return std::make_shared<Model>(OutputVector{out}, params);
}

// x * (Convert(weight) * Convert(scale))
std::shared_ptr<Model> getConvertScaleModel(element::Type weight_type,
                                            double weight_value,
                                            element::Type scale_type,
                                            double scale_value) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weight = std::make_shared<op::v0::Convert>(
        op::v0::Constant::create(weight_type, Shape{1}, std::vector<double>{weight_value}),
        element::f32);
    auto scale = std::make_shared<op::v0::Convert>(
        op::v0::Constant::create(scale_type, Shape{}, std::vector<double>{scale_value}),
        element::f32);
    auto dq = std::make_shared<op::v1::Multiply>(weight, scale);
    auto out = std::make_shared<op::v1::Multiply>(x, dq);
    return std::make_shared<Model>(OutputVector{out}, ParameterVector{x});
}

// x * Const(value)
std::shared_ptr<Model> getModelRef(const Shape& dq_shape, float value) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});
    auto dq = op::v0::Constant::create(element::f32, dq_shape, {value});
    auto out = std::make_shared<op::v1::Multiply>(x, dq);
    return std::make_shared<Model>(OutputVector{out}, ParameterVector{x});
}

class FoldScalarDQTests : public TransformationTestsF {
public:
    FoldScalarDQTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<npuw::FoldScalarDQ>();
    }
};

using FoldScalarDQParams = std::tuple<bool, bool, bool, bool, bool, Shape>;

class FoldScalarDQParamTests : public FoldScalarDQTests, public ::testing::WithParamInterface<FoldScalarDQParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<FoldScalarDQParams>& info) {
        const auto& [with_subtract, swap_subtract, zp_convert, swap_multiply, scale_convert, weight_shape] = info.param;
        std::string name = with_subtract ? "WithZP" : "NoZP";
        name += swap_subtract ? "_SwappedSub" : "";
        name += zp_convert ? "" : "_ZPNoConvert";
        name += swap_multiply ? "_SwappedMul" : "";
        name += scale_convert ? "_ScaleConvert" : "";
        name += "_rank" + std::to_string(weight_shape.size());
        return name;
    }

protected:
    void SetUp() override {
        FoldScalarDQTests::SetUp();
        DQParams p;
        std::tie(p.with_subtract, p.swap_subtract, p.zp_convert, p.swap_multiply, p.scale_convert, p.weight_shape) =
            GetParam();
        model = getModel(p);
        model_ref = getModelRef(p.weight_shape, expected_value(p));
    }
};

TEST_P(FoldScalarDQParamTests, FoldsChain) {}

INSTANTIATE_TEST_SUITE_P(NPUW,
                         FoldScalarDQParamTests,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(Shape{}, Shape{1}, Shape{1, 1})),
                         FoldScalarDQParamTests::getTestCaseName);

TEST_F(FoldScalarDQTests, NotAppliedForParameterWeight) {
    DQParams p;
    p.weight_parameter = true;
    model = getModel(p);
}

TEST_F(FoldScalarDQTests, NotAppliedForMultiElementWeight) {
    DQParams p;
    p.weight_shape = Shape{4};
    model = getModel(p);
}

TEST_F(FoldScalarDQTests, NotAppliedWithoutSubtractConvert) {
    DQParams p;
    p.weight_convert = false;
    p.zp_convert = false;
    model = getModel(p);
}

TEST_F(FoldScalarDQTests, NotAppliedWithoutWeightConvertAndSubtract) {
    DQParams p;
    p.weight_convert = false;
    p.with_subtract = false;
    model = getModel(p);
}

// Bound evaluation maps a source type's max to the destination type's max; folding must use real values.
TEST_F(FoldScalarDQTests, KeepsF16MaxScale) {
    model = getConvertScaleModel(element::u16, 1.0, element::f16, 65504.0);
    model_ref = getModelRef(Shape{1}, 65504.0f);
}

TEST_F(FoldScalarDQTests, KeepsI32MaxWeight) {
    constexpr double i32_max = std::numeric_limits<int32_t>::max();
    model = getConvertScaleModel(element::i32, i32_max, element::f32, 1.0);
    model_ref = getModelRef(Shape{1}, static_cast<float>(i32_max));
}

}  // namespace
