// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_div_with_eps_to_keep_in_mixed_precision.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace testing;
using namespace std;
using namespace ov;
using namespace ov::opset10;
const float minimal_normalized_fp16 = static_cast<float>(ov::float16::from_bits(0x0400));

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);

    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithNegativeExponent) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {-1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);
        disable_fp16_compression(mul);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_PowWithPositiveExponent) {
    // graph should be left unchanged because of the positive z in pow
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto pow_exp_const = Constant::create(element::f32, Shape{1}, {1.77});
        auto pow = std::make_shared<Power>(add, pow_exp_const);
        auto mul = std::make_shared<Multiply>(input_1, pow);

        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_MinimalPatternUnchanged) {
    // if eps_value is greater than normalized_fp16_min then leave graph unchanged
    const float eps_value = 0.0001f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto input_2 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(input_2, eps_const);
        auto divide = std::make_shared<Divide>(input_1, add);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input_1, input_2});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithMax) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{}, {eps_value});
        auto max = std::make_shared<Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(max);
        auto divide = std::make_shared<Divide>(input, sqrt);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, MarkDivWithEpsToKeepInMixedPrecision_InL2NormWithSqrtAndWithAdd) {
    const float eps_value = 1.e-12f;
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);

        model = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
        manager.register_pass<pass::MarkDivWithEpsToKeepInMixedPrecision>();
        manager.run_passes(model);
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
        auto exp = Constant::create(element::f32, Shape{}, {2.f});
        auto pow = std::make_shared<Power>(input, exp);
        auto axes_const = Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ReduceSum>(pow, axes_const);
        auto eps_const = Constant::create(element::f32, Shape{1}, {eps_value});
        auto add = std::make_shared<Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<Sqrt>(add);
        auto divide = std::make_shared<Divide>(input, sqrt);
        disable_fp16_compression(divide);

        model_ref = std::make_shared<Model>(NodeVector{divide}, ParameterVector{input});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::CONST_VALUES);
    // need to compare twice to ensure that no extra nodes are marked
    FunctionsComparator::Result result = fc(model_ref, model);
    ASSERT_TRUE(result.valid);
    result = fc(model, model_ref);
    ASSERT_TRUE(result.valid);
}
