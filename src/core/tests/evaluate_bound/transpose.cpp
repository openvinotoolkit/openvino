// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/opsets/opset9.hpp"
#include "sequnce_generator.hpp"
#include "transpose_shape_inference.hpp"

namespace {
template <typename T>
std::vector<T> tensor_to_vector(const ov::Tensor& tensor) {
    std::vector<T> rc(tensor.data<T>(), tensor.data<T>() + tensor.get_size());
    return rc;
}
}  // namespace

using namespace ov;
using namespace ov::opset9;
using namespace testing;

using TestParam = std::tuple<std::vector<int32_t>, Shape>;

class TransposeEvalBoundTest : public TestWithParam<TestParam> {
protected:
    void SetUp() override {
        std::tie(axes_order, shape) = GetParam();

        std::generate_n(std::back_inserter(lower_values), shape_size(shape), SeqGen<int32_t>(-10));
        std::generate_n(std::back_inserter(upper_values), shape_size(shape), SeqGen<int32_t>(20));

        lower_v_tensor = ov::Tensor(dtype, shape, lower_values.data());
        upper_v_tensor = ov::Tensor(dtype, shape, upper_values.data());
        axes_v_tensor = ov::Tensor(dtype, Shape{axes_order.size()}, axes_order.data());

        arg = std::make_shared<Parameter>(dtype, shape);
        order = std::make_shared<Parameter>(dtype, Shape{axes_order.size()});
        transpose = std::make_shared<Transpose>(arg, order);

        // prepare result tensors for evaluation
        auto a = std::vector<int64_t>(axes_order.begin(), axes_order.end());
        result = exp_result = TensorVector{Tensor(dtype, op::v1::calc_output_shape(transpose.get(), shape, a))};
    }

    void node_set_lower_and_upper(Node* node, const ov::Tensor& lower, const ov::Tensor& upper) {
        if (lower) {
            node->get_output_tensor(0).set_lower_value(lower);
        }

        if (upper) {
            node->get_output_tensor(0).set_upper_value(upper);
        }
    }

    Shape shape;
    element::Type dtype{element::from<int32_t>()};
    element::Type label_dtype{element::from<label_t>()};

    std::vector<int32_t> axes_order, lower_values, upper_values;
    ov::Tensor lower_v_tensor, upper_v_tensor, axes_v_tensor;
    TensorVector result, exp_result;
    std::shared_ptr<Transpose> transpose;
    std::shared_ptr<Parameter> arg, order;

    TensorLabel labels;
    TensorLabelVector out_labels = TensorLabelVector(Transpose::OUT_COUNT);
};

INSTANTIATE_TEST_SUITE_P(evaluate_bound,
                         TransposeEvalBoundTest,
                         Values(std::make_tuple(std::vector<int32_t>{0}, Shape{4}),
                                std::make_tuple(std::vector<int32_t>{0, 1}, Shape{2, 5}),
                                std::make_tuple(std::vector<int32_t>{1, 0}, Shape{2, 5}),
                                std::make_tuple(std::vector<int32_t>{0, 1, 2}, Shape{2, 3, 1}),
                                std::make_tuple(std::vector<int32_t>{1, 2, 0}, Shape{2, 3, 1}),
                                std::make_tuple(std::vector<int32_t>{1, 3, 2, 0}, Shape{2, 3, 1, 5})),
                         PrintToStringParamName());

TEST_P(TransposeEvalBoundTest, evaluate_lower) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    const auto inputs = TensorVector{Tensor(dtype, shape, lower_values.data()),
                                     Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    const auto exp_evaluate = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_lower(result), exp_evaluate);
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalBoundTest, evaluate_lower_but_arg_lower_values_not_set) {
    node_set_lower_and_upper(arg.get(), ov::Tensor(), upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_lower(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_lower_but_order_has_no_bounds_set) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);

    ASSERT_FALSE(transpose->evaluate_lower(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_upper) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    auto inputs = TensorVector{Tensor(dtype, shape, upper_values.data()),
                               Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    transpose->evaluate(exp_result, inputs);

    ASSERT_TRUE(transpose->evaluate_upper(result));
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalBoundTest, evaluate_upper_but_arg_upper_values_not_set) {
    node_set_lower_and_upper(arg.get(), upper_v_tensor, ov::Tensor());
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_upper_but_order_has_no_bounds_set) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_label_but_empty_label_set) {
    exp_result = TensorVector{Tensor(label_dtype, exp_result.front().get_shape())};

    labels.resize(shape_size(shape), 0);
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalBoundTest, evaluate_label_but_order_has_no_bound_set) {
    exp_result = TensorVector{Tensor(label_dtype, exp_result.front().get_shape())};

    std::generate_n(std::back_inserter(labels), shape_size(shape), SeqGen<label_t>(30));
    arg->get_default_output().get_tensor().set_value_label(labels);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalBoundTest, evaluate_label) {
    exp_result = TensorVector{Tensor(label_dtype, exp_result.front().get_shape())};

    std::generate_n(std::back_inserter(labels), shape_size(shape), SeqGen<label_t>(5));
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    auto inputs = TensorVector{Tensor(label_dtype, shape, labels.data()),
                               Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};

    auto exp_eval_result = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_label(out_labels), exp_eval_result);
    ASSERT_THAT(
        out_labels[Transpose::ARG_T],
        ElementsAreArray(exp_result[Transpose::ARG_T].data<label_t>(), exp_result[Transpose::ARG_T].get_size()));
}
