// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/transpose.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gmock/gmock.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/transpose.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/opsets/opset9.hpp"
#include "sequnce_generator.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

template <element::Type_t IN_ET, element::Type_t AXIS_ET>
void test_tranpose_eval(shared_ptr<Function> fun) {
    using T = typename element_type_traits<IN_ET>::value_type;
    using T_AXIS = typename element_type_traits<AXIS_ET>::value_type;

    const std::vector<std::vector<T>> input_data{{1, 2, 3, 4, 5, 6},
                                                 {1, 2, 3, 4, 5, 6},
                                                 {1, 2, 3, 4, 5, 6},
                                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> data_shapes{{2, 3}, {2, 3}, {2, 3, 1}, {2, 2, 3}};
    const std::vector<std::vector<T_AXIS>> axes_order{{0, 1}, {1, 0}, {1, 2, 0}, {2, 1, 0}};

    std::vector<std::vector<T>> expected_results{{1, 2, 3, 4, 5, 6},
                                                 {1, 4, 2, 5, 3, 6},
                                                 {1, 4, 2, 5, 3, 6},
                                                 {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};
    std::vector<Shape> expected_result_shapes{{2, 3}, {3, 2}, {3, 1, 2}, {3, 2, 2}};

    for (size_t i = 0; i < data_shapes.size(); i++) {
        auto result_tensor = make_shared<HostTensor>(element::dynamic, PartialShape::dynamic());
        auto axes_shape = axes_order[i].size() ? Shape{axes_order[i].size()} : Shape{};
        ASSERT_TRUE(fun->evaluate({result_tensor},
                                  {make_host_tensor<IN_ET>(data_shapes[i], input_data[i]),
                                   make_host_tensor<AXIS_ET>(axes_shape, axes_order[i])}));

        auto actual_results = read_vector<T>(result_tensor);
        ASSERT_EQ(actual_results, expected_results[i]);
    }
}

TEST(op_eval, eval_transpose) {
    vector<shared_ptr<op::Parameter>> axes;
    axes.push_back(make_shared<op::Parameter>(element::i8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()}));

    axes.push_back(make_shared<op::Parameter>(element::u8, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u16, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u32, PartialShape{Dimension::dynamic()}));
    axes.push_back(make_shared<op::Parameter>(element::u64, PartialShape{Dimension::dynamic()}));

    for (auto& axis : axes) {
        const auto input_integral = make_shared<op::Parameter>(element::i16, PartialShape::dynamic());
        const auto transpose_integral = make_shared<op::v1::Transpose>(input_integral, axis);
        const auto function_integral =
            make_shared<Function>(OutputVector{transpose_integral}, ParameterVector{input_integral, axis});

        const auto input_floating = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
        const auto transpose_floating = make_shared<op::v1::Transpose>(input_floating, axis);
        const auto function_floating =
            make_shared<Function>(OutputVector{transpose_floating}, ParameterVector{input_floating, axis});

        switch (axis->get_element_type()) {
        case element::Type_t::i8:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::i8>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::i8>(function_floating);
            break;
        case element::Type_t::i16:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::i16>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::i16>(function_floating);
            break;
        case element::Type_t::i32:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::i32>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::i32>(function_floating);
            break;
        case element::Type_t::i64:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::i64>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::i64>(function_floating);
            break;
        case element::Type_t::u8:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::u8>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::u8>(function_floating);
            break;
        case element::Type_t::u16:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::u16>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::u16>(function_floating);
            break;
        case element::Type_t::u32:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::u32>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::u32>(function_floating);
            break;
        case element::Type_t::u64:
            test_tranpose_eval<element::Type_t::i16, element::Type_t::u64>(function_integral);
            test_tranpose_eval<element::Type_t::f32, element::Type_t::u64>(function_floating);
            break;
        default:
            NGRAPH_CHECK(false, "Invalid type");
            break;
        }
    }
}

TEST(op_eval, eval_axes_transpose) {
    auto data_param = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{1, 2, 0};
    std::vector<int32_t> expected_result{1, 4, 2, 5, 3, 6};

    auto result_tensor = make_shared<HostTensor>();
    function->evaluate({result_tensor},
                       {make_host_tensor<element::Type_t::i32>(data_shape, data),
                        make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

    auto actual_results = read_vector<int32_t>(result_tensor);
    ASSERT_EQ(actual_results, expected_result);
}

TEST(op_eval, eval_duplicated_axes_transpose) {
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{2, 1, 2};
    try {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                           {make_host_tensor<element::Type_t::f32>(data_shape, data),
                            make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        FAIL() << "Duplicated axes values not detected";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Permutation AxisVector{2, 1, 2} is not valid for input shape"));
    } catch (...) {
        FAIL() << "Failed for unexpected reason";
    }
}

TEST(op_eval, eval_out_of_shape_axes_transpose) {
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{0, 1, 3};

    try {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                           {make_host_tensor<element::Type_t::f32>(data_shape, data),
                            make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        FAIL() << "Out of shape axes not detected";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Permutation AxisVector{0, 1, 3} is not valid for input shape"));
    } catch (...) {
        FAIL() << "Failed for unexpected reason";
    }
}

TEST(op_eval, eval_negative_axes_transpose) {
    auto data_param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes_order = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto x_transpose = make_shared<op::v1::Transpose>(data_param, axes_order);
    auto function = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{data_param, axes_order});

    const std::vector<float> data{1, 2, 3, 4, 5, 6};
    std::vector<size_t> data_shape{2, 3, 1};
    const std::vector<int32_t> perm{-1, -2, -3};
    std::vector<float> expected_result{1, 4, 2, 5, 3, 6};

    try {
        auto result_tensor = make_shared<HostTensor>();
        function->evaluate({result_tensor},
                           {make_host_tensor<element::Type_t::f32>(data_shape, data),
                            make_host_tensor<element::Type_t::i32>(Shape{perm.size()}, perm)});

        auto actual_results = read_vector<float>(result_tensor);

        ASSERT_EQ(actual_results, expected_result);
        FAIL() << "Negative axes for Transpose were not supported before.";
    } catch (const ngraph_error& error) {
        std::stringstream exp_msg;
        exp_msg << "Permutation " << AxisVector(perm.begin(), perm.end()) << " is not valid for input shape";
        EXPECT_HAS_SUBSTRING(error.what(), exp_msg.str());
    } catch (...) {
        FAIL() << "Failed for unexpected reason";
    }
}

template <typename T>
std::vector<T> tensor_to_vector(const ov::Tensor& tensor) {
    std::vector<T> rc(tensor.data<T>(), tensor.data<T>() + tensor.get_size());
    return rc;
}

using namespace ov::opset9;
using namespace testing;

using test_param = std::tuple<std::vector<int32_t>, PartialShape>;

class TransposeEvalTest : public TestWithParam<test_param> {
protected:
    void SetUp() override {
        std::tie(axes_order, p_shape) = GetParam();

        std::generate_n(std::back_inserter(lower_values),
                        ov::shape_size(p_shape.get_min_shape()),
                        ov::SeqGen<int32_t>(-10));
        std::generate_n(std::back_inserter(upper_values),
                        ov::shape_size(p_shape.get_min_shape()),
                        ov::SeqGen<int32_t>(20));

        lower_v_tensor = std::make_shared<ov::HostTensor>(dtype, p_shape.get_min_shape(), lower_values.data());
        upper_v_tensor = std::make_shared<ov::HostTensor>(dtype, p_shape.get_min_shape(), upper_values.data());
        axes_v_tensor = std::make_shared<ov::HostTensor>(dtype, Shape{axes_order.size()}, axes_order.data());

        arg = make_shared<Parameter>(dtype, p_shape);
        order = make_shared<Parameter>(dtype, Shape{axes_order.size()});
        transpose = make_shared<Transpose>(arg, order);

        // prepare result tensors for evaluation
        result = exp_result = ov::TensorVector{ov::Tensor(dtype, {0})};
    }

    void node_set_lower_and_upper(ov::Node* node, const HostTensorPtr& lower, const HostTensorPtr& upper) {
        if (lower != nullptr) {
            node->get_output_tensor(0).set_lower_value(lower);
        }

        if (upper != nullptr) {
            node->get_output_tensor(0).set_upper_value(upper);
        }
    }

    PartialShape p_shape;
    ov::element::Type dtype{ov::element::from<int32_t>()};
    ov::element::Type label_dtype{ov::element::u64};

    std::vector<int32_t> axes_order, lower_values, upper_values;
    HostTensorPtr lower_v_tensor, upper_v_tensor, axes_v_tensor;
    ov::TensorVector result, exp_result;
    std::shared_ptr<Transpose> transpose;
    std::shared_ptr<Parameter> arg, order;

    TensorLabel labels;
    TensorLabelVector out_labels = TensorLabelVector(Transpose::OUT_COUNT);
};

INSTANTIATE_TEST_SUITE_P(op_eval,
                         TransposeEvalTest,
                         Values(make_tuple(std::vector<int32_t>{0}, PartialShape{4}),
                                make_tuple(std::vector<int32_t>{0, 1}, PartialShape{2, 5}),
                                make_tuple(std::vector<int32_t>{1, 0}, PartialShape{2, 5}),
                                make_tuple(std::vector<int32_t>{0, 1, 2}, PartialShape{2, 3, 1}),
                                make_tuple(std::vector<int32_t>{1, 2, 0}, PartialShape{2, 3, 1}),
                                make_tuple(std::vector<int32_t>{1, 3, 2, 0}, PartialShape{2, 3, 1, 5})),
                         PrintToStringParamName());

TEST_P(TransposeEvalTest, evaluate_lower) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    const auto inputs = ov::TensorVector{ov::Tensor(dtype, p_shape.get_min_shape(), lower_values.data()),
                                         ov::Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    const auto exp_evaluate = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_lower(result), exp_evaluate);
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalTest, evaluate_lower_but_arg_lower_values_not_set) {
    node_set_lower_and_upper(arg.get(), nullptr, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_lower(result));
}

TEST_P(TransposeEvalTest, evaluate_lower_but_order_has_no_bounds_set) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);

    ASSERT_FALSE(transpose->evaluate_lower(result));
}

TEST_P(TransposeEvalTest, evaluate_upper) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    auto inputs = ov::TensorVector{ov::Tensor(dtype, p_shape.get_min_shape(), upper_values.data()),
                                   ov::Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};
    // evaluate expected values
    transpose->evaluate(exp_result, inputs);

    ASSERT_TRUE(transpose->evaluate_upper(result));
    ASSERT_EQ(tensor_to_vector<int32_t>(result[Transpose::ARG_T]),
              tensor_to_vector<int32_t>(exp_result[Transpose::ARG_T]));
}

TEST_P(TransposeEvalTest, evaluate_upper_but_arg_upper_values_not_set) {
    node_set_lower_and_upper(arg.get(), upper_v_tensor, nullptr);
    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalTest, evaluate_upper_but_order_has_no_bounds_set) {
    node_set_lower_and_upper(arg.get(), lower_v_tensor, upper_v_tensor);

    ASSERT_FALSE(transpose->evaluate_upper(result));
}

TEST_P(TransposeEvalTest, evaluate_label_but_empty_label_set) {
    exp_result = ov::TensorVector{ov::Tensor(label_dtype, {0})};

    labels.resize(ov::shape_size(p_shape.get_shape()), 0);
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalTest, evaluate_label_but_order_has_no_bound_set) {
    exp_result = ov::TensorVector{ov::Tensor(label_dtype, {0})};

    std::generate_n(std::back_inserter(labels), ov::shape_size(p_shape.get_shape()), ov::SeqGen<size_t>(30));
    arg->get_default_output().get_tensor().set_value_label(labels);

    ASSERT_FALSE(transpose->evaluate_label(out_labels));
}

TEST_P(TransposeEvalTest, evaluate_label) {
    exp_result = ov::TensorVector{ov::Tensor(label_dtype, {0})};

    std::generate_n(std::back_inserter(labels), ov::shape_size(p_shape.get_shape()), ov::SeqGen<size_t>(5));
    arg->get_default_output().get_tensor().set_value_label(labels);

    node_set_lower_and_upper(order.get(), axes_v_tensor, axes_v_tensor);

    auto labels_u64 = std::vector<uint64_t>(labels.cbegin(), labels.cend());
    auto inputs = ov::TensorVector{ov::Tensor(label_dtype, p_shape.get_shape(), labels_u64.data()),
                                   ov::Tensor(dtype, Shape{axes_order.size()}, axes_order.data())};

    auto exp_eval_result = transpose->evaluate(exp_result, inputs);

    ASSERT_EQ(transpose->evaluate_label(out_labels), exp_eval_result);
    ASSERT_THAT(
        out_labels[Transpose::ARG_T],
        ElementsAreArray(exp_result[Transpose::ARG_T].data<uint64_t>(), exp_result[Transpose::ARG_T].get_size()));
}
