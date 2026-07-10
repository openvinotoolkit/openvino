// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/strided_slice_reshape_concat_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;

class StridedSliceReshapeConcatFusionTest : public TransformationTestsF {};

TEST_F(StridedSliceReshapeConcatFusionTest, Positive) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
        ov::OutputVector concat_inputs;

        for (const int64_t start : {0, 2, 4}) {
            auto begin = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {0, start});
            auto end = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, start + 4});
            auto strides = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
            auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                                            begin,
                                                                            end,
                                                                            strides,
                                                                            std::vector<int64_t>{0, 0},
                                                                            std::vector<int64_t>{0, 0});
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 1, 4});
            auto reshape = std::make_shared<ov::op::v1::Reshape>(strided_slice, shape, false);
            concat_inputs.push_back(reshape);
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::StridedSliceReshapeConcatFusion>();
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
        auto indices = ov::op::v0::Constant::create<int64_t>(ov::element::i64,
                                                             ov::Shape{3, 4},
                                                             {0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto gather = std::make_shared<ov::op::v8::Gather>(input, indices, axis, 0);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{input});
    }
}

TEST_F(StridedSliceReshapeConcatFusionTest, NegativeUnequalSliceLength) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    ov::OutputVector concat_inputs;

    for (const auto& range : std::vector<std::pair<int64_t, int64_t>>{{0, 4}, {4, 9}, {9, 13}}) {
        const auto start = range.first;
        const auto stop = range.second;
        auto begin = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {0, start});
        auto end = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, stop});
        auto strides = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                                        begin,
                                                                        end,
                                                                        strides,
                                                                        std::vector<int64_t>{0, 0},
                                                                        std::vector<int64_t>{0, 0});
        auto shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 1, stop - start});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(strided_slice, shape, false);
        concat_inputs.push_back(reshape);
    }

    // Unequal slice lengths require concatenation over width dimension to keep graph valid.
    auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 2);
    model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
    model_ref = model->clone();

    manager.register_pass<ov::pass::StridedSliceReshapeConcatFusion>();
}

TEST(StridedSliceReshapeConcatFusionInferTest, NumericalEquivalence) {
    auto build_original = []() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
        ov::OutputVector concat_inputs;

        for (const int64_t start : {0, 2, 4}) {
            auto begin = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {0, start});
            auto end = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, start + 4});
            auto strides = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 1});
            auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                                            begin,
                                                                            end,
                                                                            strides,
                                                                            std::vector<int64_t>{0, 0},
                                                                            std::vector<int64_t>{0, 0});
            auto shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 1, 4});
            auto reshape = std::make_shared<ov::op::v1::Reshape>(strided_slice, shape, false);
            concat_inputs.push_back(reshape);
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
        return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
    };

    auto original = build_original();
    auto transformed = original->clone();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::StridedSliceReshapeConcatFusion>();
    manager.run_passes(transformed);

    const auto input_tensor =
        ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                ov::Shape{1, 16},
                                                ov::test::utils::InputGenerateData{-3.0, 7, 1, 1});
    ov::TensorVector inputs{input_tensor};

    const auto outputs_before = ov::test::utils::infer_on_template(original, inputs);
    const auto outputs_after = ov::test::utils::infer_on_template(transformed, inputs);

    ASSERT_EQ(outputs_before.size(), outputs_after.size());
    ASSERT_EQ(outputs_before.size(), 1);
    ov::test::utils::compare(outputs_before[0], outputs_after[0], 0.0, 0.0);
}
