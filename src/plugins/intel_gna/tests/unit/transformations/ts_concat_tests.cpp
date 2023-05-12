// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/ts_concat.hpp>
#include <transformations/ts_split.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "debug_new_pass.hpp"
#include "gtest/gtest.h"
#include "ngraph/pass/visualize_tree.hpp"

using namespace ov;
using namespace ov::opset10;

namespace {

std::shared_ptr<void> GenerateFloatInput(size_t size, float initial_value, float delta) {
    float* array = new float[size];
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
    ss << "[EMUTEX ASSERT] " << first_name << " (" << first << ") != " << second_name << "(" << second << ")";
    throw std::runtime_error(ss.str());
}

#define EMUTEX_DEBUG_ASSERT_EQ(first, second) AssertEq(first, second, #first, #second);
template <typename T, typename T1>
void AssertEqPrecision(const T& first,
                       const T& second,
                       const T1& delta,
                       const std::string& first_name,
                       const std::string& second_name) {
    if (std::abs(first - second) <= delta)
        return;
    std::ostringstream ss;
    ss << "[EMUTEX ASSERT] " << first_name << " (" << first << ")  != " << second_name << " (" << second
       << ") with precision " << delta;
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
    const size_t input_shape_product =
        std::accumulate(function_input_shape.begin(), function_input_shape.end(), 1, std::multiplies<size_t>());
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
        const float* result_data = result[output_idx].data<float>();
        const float* expected_result = result_ref[output_idx].data<float>();
        for (size_t i = 0; i < result[output_idx].get_size(); ++i) {
            EMUTEX_DEBUG_ASSERT_EQ_PRECISION(result_data[i], expected_result[i], 0.000001);
        }
    }
}

}  // namespace

TEST(TSConcat, Forward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 9, 22});

        auto split_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 3);
        auto split = std::make_shared<Split>(input_params, split_const, 2);

        ov::OutputVector split_outputs;
        for (auto& output : split->outputs()) {
            auto transpose_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<Transpose>(output, transpose_const);
            split_outputs.push_back(transpose->output(0));
        }

        auto concat = std::make_shared<Concat>(split_outputs, 1);

        const auto result = std::make_shared<Result>(concat);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    EMUTEX_DEBUG_VISUALIZE("before");
    manager.register_pass<ov::intel_gna::pass::TSConcatForward>();
    EMUTEX_DEBUG_VISUALIZE("after");
    manager.run_passes(function);

    CompareOutput(orig_function, function);
}

TEST(TSSplit, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 22, 9});

        auto split_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 2);
        auto split = std::make_shared<Split>(input_params, split_const, 2);

        ov::OutputVector split_outputs;
        for (auto& output : split->outputs()) {
            auto transpose_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<Transpose>(output, transpose_const);
            split_outputs.push_back(transpose->output(0));
        }

        auto concat = std::make_shared<Concat>(split_outputs, 1);

        const auto result = std::make_shared<Result>(concat);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    EMUTEX_DEBUG_VISUALIZE("before");
    manager.register_pass<ov::intel_gna::pass::TSSplitBackward>();
    EMUTEX_DEBUG_VISUALIZE("after");
    manager.run_passes(function);

    CompareOutput(orig_function, function);
}
