// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_concat_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

namespace v0 = ov::op::v0;

namespace {

std::shared_ptr<v0::FakeQuantize> make_fake_quantize(const Output<Node>& data,
                                                     float input_low,
                                                     float input_high,
                                                     float output_low,
                                                     float output_high,
                                                     size_t levels) {
    const auto in_low = v0::Constant::create(element::f32, Shape{1}, {input_low});
    const auto in_high = v0::Constant::create(element::f32, Shape{1}, {input_high});
    const auto out_low = v0::Constant::create(element::f32, Shape{1}, {output_low});
    const auto out_high = v0::Constant::create(element::f32, Shape{1}, {output_high});
    return std::make_shared<v0::FakeQuantize>(data, in_low, in_high, out_low, out_high, levels);
}

}  // namespace

TEST_F(TransformationTestsF, FakeQuantizeConcatFusion_eliminates_redundant_input_fqs) {
    {
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq0 = make_fake_quantize(input0, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq1 = make_fake_quantize(input1, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto concat = std::make_shared<v0::Concat>(OutputVector{fq0, fq1}, 1);
        auto output_fq = make_fake_quantize(concat, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(output_fq);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1});
        manager.register_pass<ov::pass::FakeQuantizeConcatFusion>();
    }
    {
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto concat = std::make_shared<v0::Concat>(OutputVector{input0, input1}, 1);
        auto output_fq = make_fake_quantize(concat, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(output_fq);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1});
    }
}

TEST_F(TransformationTestsF, FakeQuantizeConcatFusion_not_applied_when_input_fqs_differ) {
    {
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq0 = make_fake_quantize(input0, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq1 = make_fake_quantize(input1, -0.5f, 0.5f, -1.0f, 1.0f, 256);
        auto concat = std::make_shared<v0::Concat>(OutputVector{fq0, fq1}, 1);
        auto output_fq = make_fake_quantize(concat, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(output_fq);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1});
        manager.register_pass<ov::pass::FakeQuantizeConcatFusion>();
    }

    model_ref = model->clone();
}

TEST_F(TransformationTestsF, FakeQuantizeConcatFusion_not_applied_when_output_fq_differs) {
    {
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq0 = make_fake_quantize(input0, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq1 = make_fake_quantize(input1, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto concat = std::make_shared<v0::Concat>(OutputVector{fq0, fq1}, 1);
        auto output_fq = make_fake_quantize(concat, -2.0f, 2.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(output_fq);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1});
        manager.register_pass<ov::pass::FakeQuantizeConcatFusion>();
    }

    model_ref = model->clone();
}

TEST_F(TransformationTestsF, FakeQuantizeConcatFusion_cascaded_fusion) {
    {
        // First pattern: FQ0, FQ1 -> Concat1 -> FQ2
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq0 = make_fake_quantize(input0, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq1 = make_fake_quantize(input1, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto concat1 = std::make_shared<v0::Concat>(OutputVector{fq0, fq1}, 1);
        auto fq2 = make_fake_quantize(concat1, -1.0f, 1.0f, -1.0f, 1.0f, 256);

        // Second pattern: FQ3, FQ4 -> Concat2 -> FQ5
        auto input2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input3 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq3 = make_fake_quantize(input2, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq4 = make_fake_quantize(input3, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto concat2 = std::make_shared<v0::Concat>(OutputVector{fq3, fq4}, 1);
        auto fq5 = make_fake_quantize(concat2, -1.0f, 1.0f, -1.0f, 1.0f, 256);

        // Third level: Concat (FQ2, FQ5) -> FQ6
        auto concat3 = std::make_shared<v0::Concat>(OutputVector{fq2, fq5}, 1);
        auto fq6 = make_fake_quantize(concat3, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq6);

        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1, input2, input3});
        manager.register_pass<ov::pass::FakeQuantizeConcatFusion>();
    }
    {
        // After cascaded fusion, all three FQ->Concat->FQ patterns are fused
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto concat1 = std::make_shared<v0::Concat>(OutputVector{input0, input1}, 1);

        auto input2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input3 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto concat2 = std::make_shared<v0::Concat>(OutputVector{input2, input3}, 1);

        // Third level: concat1 and concat2 are directly concatenated and FQed
        auto concat3 = std::make_shared<v0::Concat>(OutputVector{concat1, concat2}, 1);
        auto fq6 = make_fake_quantize(concat3, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq6);

        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1, input2, input3});
    }
}

TEST_F(TransformationTestsF, FakeQuantizeConcatFusion_not_applied_when_mixed_fq_and_non_fq_inputs) {
    {
        // One input is FQ, the other is not
        auto input0 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto input1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto fq0 = make_fake_quantize(input0, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto concat = std::make_shared<v0::Concat>(OutputVector{fq0, input1}, 1);
        auto output_fq = make_fake_quantize(concat, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(output_fq);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input0, input1});
        manager.register_pass<ov::pass::FakeQuantizeConcatFusion>();
    }

    model_ref = model->clone();
}
