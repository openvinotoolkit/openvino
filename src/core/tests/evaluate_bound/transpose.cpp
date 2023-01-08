// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "gmock/gmock.h"
#include "openvino/opsets/opset9.hpp"
#include "sequnce_generator.hpp"

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

using TestParam = std::tuple<std::vector<int32_t>, PartialShape>;

class TransposeEvalBoundTest : public TestWithParam<TestParam> {
protected:
    void SetUp() override {
        std::tie(axes_order, p_shape) = GetParam();

        std::generate_n(std::back_inserter(lower_values), shape_size(p_shape.get_min_shape()), SeqGen<int32_t>(-10));
        std::generate_n(std::back_inserter(upper_values), shape_size(p_shape.get_min_shape()), SeqGen<int32_t>(20));

        lower_v_tensor = std::make_shared<HostTensor>(dtype, p_shape.get_min_shape(), lower_values.data());
        upper_v_tensor = std::make_shared<HostTensor>(dtype, p_shape.get_min_shape(), upper_values.data());
        axes_v_tensor = std::make_shared<HostTensor>(dtype, Shape{axes_order.size()}, axes_order.data());

        arg = std::make_shared<Parameter>(dtype, p_shape);
        order = std::make_shared<Parameter>(dtype, Shape{axes_order.size()});
        transpose = std::make_shared<Transpose>(arg, order);

        // prepare result tensors for evaluation
        result = exp_result = TensorVector{Tensor(dtype, {0})};
    }

    void node_set_lower_and_upper(Node* node, const HostTensorPtr& lower, const HostTensorPtr& upper) {
        if (lower != nullptr) {
            node->get_output_tensor(0).set_lower_value(lower);
        }

        if (upper != nullptr) {
            node->get_output_tensor(0).set_upper_value(upper);
        }
    }

    PartialShape p_shape;
    element::Type dtype{element::from<int32_t>()};
    element::Type label_dtype{element::u64};

    std::vector<int32_t> axes_order, lower_values, upper_values;
    HostTensorPtr lower_v_tensor, upper_v_tensor, axes_v_tensor;
    TensorVector result, exp_result;
    std::shared_ptr<Transpose> transpose;
    std::shared_ptr<Parameter> arg, order;

    TensorLabel labels;
    TensorLabelVector out_labels = TensorLabelVector(Transpose::OUT_COUNT);
};

INSTANTIATE_TEST_SUITE_P(evaluate_bound,
                         TransposeEvalBoundTest,
                         Values(std::make_tuple(std::vector<int32_t>{0}, PartialShape{4}),
                                std::make_tuple(std::vector<int32_t>{0, 1}, PartialShape{2, 5}),
                                std::make_tuple(std::vector<int32_t>{1, 0}, PartialShape{2, 5}),
                                std::make_tuple(std::vector<int32_t>{0, 1, 2}, PartialShape{2, 3, 1}),
                                std::make_tuple(std::vector<int32_t>{1, 2, 0}, PartialShape{2, 3, 1}),
                                std::make_tuple(std::vector<int32_t>{1, 3, 2, 0}, PartialShape{2, 3, 1, 5})),
                         PrintToStringParamName());

TEST_P(TransposeEvalBoundTest, evaluate_lower) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    const auto inputs = TensorVector{Tensor(dtype, p_shape.get_min_shape(), lower_values.data()),
                                     Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    const auto exp_evaluate = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_lower(result), exp_evaluate);
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalBoundTest, evaluate_lower_but_arg_lower_values_not_set) {
    node_set_lower_and_upper(arg.get(), nullptr, upper_v_tensor);
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

    auto inputs = TensorVector{Tensor(dtype, p_shape.get_min_shape(), upper_values.data()),
                               Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    transpose->evaluate(exp_result, inputs);

    ASSERT_TRUE(transpose->evaluate_upper(result));
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalBoundTest, evaluate_upper_but_arg_upper_values_not_set) {
    node_set_lower_and_upper(arg.get(), upper_v_tensor, nullptr);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_upper_but_order_has_no_bounds_set) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalBoundTest, evaluate_label_but_empty_label_set) {
    exp_result = TensorVector{Tensor(label_dtype, {0})};

    labels.resize(shape_size(p_shape.get_shape()), 0);
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalBoundTest, evaluate_label_but_order_has_no_bound_set) {
    exp_result = TensorVector{Tensor(label_dtype, {0})};

    std::generate_n(std::back_inserter(labels), shape_size(p_shape.get_shape()), SeqGen<size_t>(30));
    arg->get_default_output().get_tensor().set_value_label(labels);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalBoundTest, evaluate_label) {
    exp_result = TensorVector{Tensor(label_dtype, {0})};

    std::generate_n(std::back_inserter(labels), shape_size(p_shape.get_shape()), SeqGen<size_t>(5));
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    auto labels_u64 = std::vector<uint64_t>(labels.cbegin(), labels.cend());
    auto inputs = TensorVector{Tensor(label_dtype, p_shape.get_shape(), labels_u64.data()),
                               Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};

    auto exp_eval_result = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_label(out_labels), exp_eval_result);
    ASSERT_THAT(
        out_labels[Transpose::ARG_T],
        ElementsAreArray(exp_result[Transpose::ARG_T].data<uint64_t>(), exp_result[Transpose::ARG_T].get_size()));
}
