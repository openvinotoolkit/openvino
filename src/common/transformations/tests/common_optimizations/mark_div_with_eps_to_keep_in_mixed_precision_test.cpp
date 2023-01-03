// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/opsets/opset4.hpp>
#include <openvino/pass/manager.hpp>
#include <string>
#include "transformations/common_optimizations/mark_div_with_eps_to_keep_in_mixed_precision.hpp"
#include <transformations/init_node_info.hpp>
#include "transformations/rt_info/disable_fp16_compression.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace std;
using namespace ov;
const float minimal_normalized_fp16 = static_cast<float>(ov::float16::from_bits(0x0400));

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithNegativeExponent) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);
        disable_fp16_compression(mul);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithPositiveExponent) {
    // graph should be left unchanged because of the positive z in pow
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto pow_exp_const = opset4::Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<opset4::Power>(add, pow_exp_const);
        auto mul = std::make_shared<opset4::Multiply>(input_1, pow);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_MinimalPatternUnchanged) {
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(input_2, eps_const);
        auto divide = std::make_shared<opset4::Divide>(input_1, add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithMax) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(max);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithAdd) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = opset4::Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<opset4::Power>(input, exp);
        auto axes_const = opset4::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<opset4::ReduceSum>(pow, axes_const);
        auto eps_const = opset4::Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<opset4::Sqrt>(add);
        auto divide = std::make_shared<opset4::Divide>(input, sqrt);
        disable_fp16_compression(divide);
        
        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::RUNTIME_KEYS)
            .enable(FunctionsComparator::CONST_VALUES);
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    // need to compare twice ensure that there are no extra runtime keys in model_ref
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}
