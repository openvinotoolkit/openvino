// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "utils.hpp"
using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using matmul_test_params_t = std::tuple<StaticShape,  // Input A shape
                                        StaticShape   // Input B shape
                                        >;

class MatMulTest : public TestWithParam<matmul_test_params_t> {
protected:
    void SetUp() override {
        std::tie(a_shape, b_shape) = GetParam();

        set_exp_shape();
    }

    std::shared_ptr<op::v0::MatMul> make_matmul(const size_t& a_dim_count,
                                                const size_t& b_dim_count,
                                                const bool transpose_a,
                                                const bool transpose_b) {
        auto a_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(a_dim_count));
        auto b_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(b_dim_count));

        return std::make_shared<op::v0::MatMul>(a_input, b_input, transpose_a, transpose_b);
    }

    void set_exp_shape() {
        if (a_shape.size() > 1 && b_shape.size() > 1) {
            std::transform(a_shape.cbegin(),
                           a_shape.cend() - 2,
                           b_shape.cbegin(),
                           std::back_inserter(exp_shape),
                           [](const StaticDimension& a, const StaticDimension& b) {
                               return std::max(a.get_length(), b.get_length());
                           });
            exp_shape.push_back(*std::next((*a_shape).rbegin()));
            exp_shape.push_back((*b_shape).back());
        } else if (a_shape.size() == 1 && b_shape.size() > 1) {
            exp_shape = b_shape;
            (*exp_shape).erase(std::prev((*exp_shape).end(), 2));
        } else if (b_shape.size() == 1 && a_shape.size() > 1) {
            exp_shape = a_shape;
            (*exp_shape).erase(std::prev((*exp_shape).end()));
        }
    }

    static StaticShape make_transpose_input(const StaticShape& in) {
        StaticShape out(in);
        if (out.size() > 1) {
            std::iter_swap((*out).rbegin(), std::next((*out).rbegin()));
        }
        return out;
    }

    StaticShape a_shape, b_shape, exp_shape;
};

/** \brief Use transpose order -> output shape dimensions shall be as transpose order. */
INSTANTIATE_TEST_SUITE_P(StaticShapeInference,
                         MatMulTest,
                         Values(make_tuple(StaticShape({1}), StaticShape({1})),
                                make_tuple(StaticShape({1}), StaticShape({1, 3})),
                                make_tuple(StaticShape({1}), StaticShape({1, 1, 3})),
                                make_tuple(StaticShape({3, 1}), StaticShape({1})),
                                make_tuple(StaticShape({3, 2, 1}), StaticShape({1})),
                                make_tuple(StaticShape({3}), StaticShape({3})),
                                make_tuple(StaticShape({5, 2}), StaticShape({2, 6})),
                                make_tuple(StaticShape({2, 1, 2}), StaticShape({2, 6})),
                                make_tuple(StaticShape({10, 8, 9, 2}), StaticShape({10, 8, 2, 8})),
                                make_tuple(StaticShape({3, 1, 4, 3, 4}), StaticShape({3, 2, 1, 4, 1}))),
                         PrintToStringParamName());

TEST_P(MatMulTest, no_input_transpose) {
    const auto matmul = make_matmul(a_shape.size(), b_shape.size(), false, false);

    std::vector<StaticShape> static_input_shapes = {a_shape, b_shape}, static_output_shapes = {StaticShape{}};

    static_output_shapes = shape_inference(matmul.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.front(), exp_shape);
}

TEST_P(MatMulTest, transpose_input_a) {
    const auto matmul = make_matmul(a_shape.size(), b_shape.size(), true, false);

    const auto a_transpose = make_transpose_input(a_shape);
    std::vector<StaticShape> static_input_shapes = {a_transpose, b_shape}, static_output_shapes = {StaticShape{}};

    static_output_shapes = shape_inference(matmul.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.front(), exp_shape);
}

TEST_P(MatMulTest, transpose_input_b) {
    const auto matmul = make_matmul(a_shape.size(), b_shape.size(), false, true);

    const auto b_transpose = make_transpose_input(b_shape);
    std::vector<StaticShape> static_input_shapes = {a_shape, b_transpose}, static_output_shapes = {StaticShape{}};

    static_output_shapes = shape_inference(matmul.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.front(), exp_shape);
}

TEST_P(MatMulTest, transpose_inputs_a_b) {
    const auto matmul = make_matmul(a_shape.size(), b_shape.size(), true, true);

    const auto a_transpose = make_transpose_input(a_shape);
    const auto b_transpose = make_transpose_input(b_shape);

    std::vector<StaticShape> static_input_shapes = {a_transpose, b_transpose}, static_output_shapes = {StaticShape{}};

    static_output_shapes = shape_inference(matmul.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.front(), exp_shape);
}
