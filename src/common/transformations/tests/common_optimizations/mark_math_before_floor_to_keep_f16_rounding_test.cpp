// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/mark_math_before_floor_to_keep_f16_rounding.hpp"

#include <functional>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/tan.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace {

using MathBuilder = std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&)>;

template <class Op>
MathBuilder unary() {
    return [](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(in);
    };
}

template <class Op>
MathBuilder with_two_constants(float first, float second) {
    return [=](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(in,
                                    ov::op::v0::Constant::create(ov::element::f16, {}, {first}),
                                    ov::op::v0::Constant::create(ov::element::f16, {}, {second}));
    };
}

const std::vector<MathBuilder> math_builders = {
    unary<ov::op::v0::Cos>(),
    unary<ov::op::v0::Cosh>(),
    unary<ov::op::v0::Sin>(),
    unary<ov::op::v0::Sinh>(),
    unary<ov::op::v0::Acos>(),
    unary<ov::op::v3::Acosh>(),
    unary<ov::op::v0::Asin>(),
    unary<ov::op::v3::Asinh>(),
    unary<ov::op::v0::Atan>(),
    unary<ov::op::v3::Atanh>(),
    unary<ov::op::v0::Tan>(),
    unary<ov::op::v0::Sign>(),
    unary<ov::op::v4::SoftPlus>(),
    unary<ov::op::v9::SoftSign>(),
    with_two_constants<ov::op::v0::Selu>(1.67326f, 1.0507f),
    with_two_constants<ov::op::v0::HardSigmoid>(0.2f, 0.5f),
};

// Builds Parameter -> Math -> Consumer; the Math node is marked when `mark_math` is set.
template <class Consumer>
std::shared_ptr<ov::Model> make_model(const MathBuilder& build_math, bool mark_math) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto math = build_math(input);
    if (mark_math) {
        ov::disable_conversion(math, ov::element::f16, ov::element::f32);
    }
    auto consumer = std::make_shared<Consumer>(math);
    return std::make_shared<ov::Model>(ov::OutputVector{consumer}, ov::ParameterVector{input});
}

class MarkMathBeforeFloorToKeepF16RoundingTest : public TransformationTestsF,
                                                 public testing::WithParamInterface<MathBuilder> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<MathBuilder>& obj) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        return obj.param(input)->get_type_name();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
        manager.register_pass<ov::pass::MarkMathBeforeFloorToKeepF16Rounding>();
    }
};

TEST_P(MarkMathBeforeFloorToKeepF16RoundingTest, MathFeedingFloorIsMarked) {
    model = make_model<ov::op::v0::Floor>(GetParam(), false);
    model_ref = make_model<ov::op::v0::Floor>(GetParam(), true);
}

TEST_P(MarkMathBeforeFloorToKeepF16RoundingTest, MathNotFeedingFloorIsNotMarked) {
    model = make_model<ov::op::v0::Relu>(GetParam(), false);
    model_ref = make_model<ov::op::v0::Relu>(GetParam(), false);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         MarkMathBeforeFloorToKeepF16RoundingTest,
                         testing::ValuesIn(math_builders),
                         MarkMathBeforeFloorToKeepF16RoundingTest::get_test_case_name);

}  // namespace
