// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

#include "ngraph/pass/visualize_tree.hpp" // DEBUG

using namespace ov;
using namespace ov::opset9;

namespace testing {

namespace {
std::shared_ptr<void> GenerateFloatInput(size_t size, float initial_value, float delta) {
    float * array = new float[size];
    float value = initial_value;
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
        value += delta;
    }
    return std::shared_ptr<void>(array);
}
template <typename T>
void AssertEq(const T& first, const T& second, const std::string& first_name, const std::string& second_name) {
    if (first == second)
        return;
    std::ostringstream ss;
    ss << "[EMUTEX ASSERT] " << first_name << " (" << first  << ") != " << second_name << "(" << second << ")";
    throw std::runtime_error(ss.str());
}
#define EMUTEX_DEBUG_ASSERT_EQ(first, second) AssertEq(first, second, #first, #second);
template <typename T, typename T1>
void AssertEqPrecision(const T& first, const T& second, const T1& delta, const std::string& first_name, const std::string& second_name) {
    if (std::abs(first - second) <= delta)
        return;
    std::ostringstream ss;
    ss << "[EMUTEX ASSERT] " << first_name << " (" << first  << ")  != " << second_name << " (" << second << ") with precision " << delta;
    throw std::runtime_error(ss.str());
}
#define EMUTEX_DEBUG_ASSERT_EQ_PRECISION(first, second, delta) AssertEqPrecision(first, second, delta, #first, #second);
void CompareOutput(std::shared_ptr<ov::Model> function, std::shared_ptr<ov::Model> function_ref) {
    auto function_input = function->input(0).get_node_shared_ptr();
    auto function_ref_input = function_ref->input(0).get_node_shared_ptr();
    const auto& function_input_shape = function_input->get_output_shape(0);
    const auto& function_ref_input_shape = function_ref_input->get_output_shape(0);
    bool rc = std::equal(function_input_shape.begin(), function_input_shape.end(), function_ref_input_shape.begin());
    if (!rc)
        throw std::runtime_error("function_input_shape != function_ref_input_shape");
    const size_t n_outputs = function->outputs().size();
    ov::TensorVector result(n_outputs), result_ref(n_outputs);
    const size_t input_shape_product = std::accumulate(function_input_shape.begin(), function_input_shape.end(), 1, std::multiplies<size_t>());
    auto inputs = GenerateFloatInput(input_shape_product, 0.0, 0.1);
    ov::Tensor input{ov::element::f32, function_input_shape, inputs.get()};
    rc = function->evaluate(result, ov::TensorVector{input});
    if (!rc)
        throw std::runtime_error("function->evaluate");
    rc = function_ref->evaluate(result_ref, ov::TensorVector{input});
    if (!rc)
        throw std::runtime_error("function_ref->evaluate");
    EMUTEX_DEBUG_ASSERT_EQ(result.size(), result_ref.size());
    for (size_t output_idx = 0; output_idx < n_outputs; ++output_idx) {
        EMUTEX_DEBUG_ASSERT_EQ(result[output_idx].get_element_type(), result_ref[output_idx].get_element_type());
        EMUTEX_DEBUG_ASSERT_EQ(result[output_idx].get_shape(), result_ref[output_idx].get_shape());
        EMUTEX_DEBUG_ASSERT_EQ(result[output_idx].get_size(), result_ref[output_idx].get_size());
        const float * result_data = result[output_idx].data<float>();
        const float * expected_result = result_ref[output_idx].data<float>();
        for (size_t i = 0; i < result[output_idx].get_size(); ++i) {
            EMUTEX_DEBUG_ASSERT_EQ_PRECISION(result_data[i], expected_result[i], 0.000001);
        }
    }
}

template <typename T>
std::shared_ptr<T> FindNode(std::shared_ptr<Model> model) {
    for (auto op : model->get_ops()) {
        auto node = as_type_ptr<T>(op);
        if (node)
            return node;
    }
    return {};
}

void PrintConstant(std::shared_ptr<Node> node) {
    auto constant = as_type_ptr<Constant>(node);
    if (!constant)
        return;
    auto value = constant->cast_vector<int>();
    std::cout << "{ ";
    for (int i = 0; i < value.size(); ++i) {
        if (i)
            std::cout << ", ";
        std::cout << value[i];
    }
    std::cout << " }" << std::endl;
}

} // namespace

TEST(GatherSinkingTransposeReshape, ForwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(tanh0, transpose_order);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png");
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeForward>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png");
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    CompareOutput(function, orig_function);

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto generate_indices = []() -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (int i = 0; i < 80; ++i) {
                indices.push_back(i);
                indices.push_back(i + 80);
                indices.push_back(i + 160);
            }
            return indices;
        };
        auto gather_indices = generate_indices();
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(reshape, gather_indices_const, gather_axis_const);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST(GatherSinkingTransposeReshape, BackwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 240});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int>{1, 3, 80});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(reshape, transpose_order);

        auto tanh1 = std::make_shared<Tanh>(transpose);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png");
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeBackward>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png");
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    CompareOutput(function, orig_function);

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 240});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto generate_indices = []() -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (int i = 0; i < 80; ++i) {
                indices.push_back(i);
                indices.push_back(i + 80);
                indices.push_back(i + 160);
            }
            return indices;
        };
        auto gather_indices = generate_indices();
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(tanh0, gather_indices_const, gather_axis_const);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int>{1, 80, 3});
        auto reshape = std::make_shared<Reshape>(gather, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingTransposeReshape, ForwardSinkingNoSink1) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(tanh0, transpose_order);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 3, 80, 1});
        auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, orig_function);
    ASSERT_TRUE(result.valid);
}

TEST(GatherSinkingTransposeReshape, ForwardSinkingNoSink2) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 4, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(tanh0, transpose_order);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 2, 80, 2});
        auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, orig_function);
    ASSERT_TRUE(result.valid);
}

TEST(GatherSinkingTransposeReshape, BackwardSinkingNoSink1) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 240});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 3, 80, 1});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 1, 3});
        auto transpose = std::make_shared<Transpose>(reshape, transpose_order);

        auto tanh1 = std::make_shared<Tanh>(transpose);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png");
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeBackward>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png");
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, orig_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingTransposeReshape, BackwardSinkingNoSink2) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 320});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 2, 80, 2});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 1, 3});
        auto transpose = std::make_shared<Transpose>(reshape, transpose_order);

        auto tanh1 = std::make_shared<Tanh>(transpose);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./0before.png");
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeBackward>();
    //manager.register_pass<ngraph::pass::VisualizeTree>("./1after.png");
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, orig_function);
    ASSERT_TRUE(result.valid) << result.message;
}

} // namespace testing
