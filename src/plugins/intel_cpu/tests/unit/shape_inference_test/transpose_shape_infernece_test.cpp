// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "transpose_shape_inference.hpp"
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using transpose_params = std::tuple<std::vector<size_t>,  // transpose order
                                    StaticShape,          // Input shape
                                    StaticShape           // Expected shape
                                    >;

class StaticShapeInferenceTest : public TestWithParam<transpose_params> {
    template <class TInput, class TOrder>
    std::shared_ptr<op::v1::Transpose> make_transpose(const TInput& input_shape, const TOrder& transpose_order) {
        const auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(input_shape.size()));
        const auto order =
            std::make_shared<op::v0::Constant>(element::i64, ov::Shape{transpose_order.size()}, transpose_order);
        return std::make_shared<op::v1::Transpose>(input, order);
    }

protected:
    void SetUp() override {
        std::tie(transpose_order, input_shape, exp_shape) = GetParam();

        transpose = make_transpose(input_shape, transpose_order);
    }

    StaticShape input_shape, exp_shape;
    std::vector<size_t> transpose_order;

    std::shared_ptr<op::v1::Transpose> transpose;
};

/** \brief Use transpose order -> output shape dimensions shall be as transpose order. */
INSTANTIATE_TEST_SUITE_P(
    transpose_by_order,
    StaticShapeInferenceTest,
    Values(make_tuple(std::vector<size_t>{0}, StaticShape({3}), StaticShape({3})),
           make_tuple(std::vector<size_t>{0, 1}, StaticShape({5, 2}), StaticShape({5, 2})),
           make_tuple(std::vector<size_t>{1, 0}, StaticShape({8, 3}), StaticShape({3, 8})),
           make_tuple(std::vector<size_t>{2, 0, 1}, StaticShape({1, 0, 2}), StaticShape({2, 1, 0})),
           make_tuple(std::vector<size_t>{2, 0, 3, 1}, StaticShape({10, 8, 9, 2}), StaticShape({9, 10, 2, 8})),
           make_tuple(std::vector<size_t>{1, 3, 2, 0}, StaticShape({1, 2, 3, 4}), StaticShape({2, 4, 3, 1}))),
    PrintToStringParamName());

/** \brief Empty transpose order -> output shape dimensions shall be in reverse order. */
INSTANTIATE_TEST_SUITE_P(
    transpose_reverse,
    StaticShapeInferenceTest,
    Values(make_tuple(std::vector<size_t>{}, StaticShape({1}), StaticShape({1})),
           make_tuple(std::vector<size_t>{}, StaticShape({23}), StaticShape({23})),
           make_tuple(std::vector<size_t>{}, StaticShape({3, 8}), StaticShape({8, 3})),
           make_tuple(std::vector<size_t>{}, StaticShape({1, 0, 2}), StaticShape({2, 0, 1})),
           make_tuple(std::vector<size_t>{}, StaticShape({21, 1, 5, 9}), StaticShape({9, 5, 1, 21})),
           make_tuple(std::vector<size_t>{}, StaticShape({0, 0, 0}), StaticShape({0, 0, 0})),
           make_tuple(std::vector<size_t>{}, StaticShape({0, 2, 0}), StaticShape({0, 2, 0})),
           make_tuple(std::vector<size_t>{}, StaticShape({0, 2, 0, 0}), StaticShape({0, 0, 2, 0}))),
    PrintToStringParamName());

/** \brief Check shape_infer for transpose on static shapes. */
TEST_P(StaticShapeInferenceTest, transpose_static) {
    auto output_shapes = std::vector<StaticShape>{StaticShape{}};

    shape_infer(transpose.get(), {input_shape, transpose_order}, output_shapes);

    ASSERT_EQ(output_shapes[op::TransposeOut::ARG_T], exp_shape);
}
