//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "interpreter_engine.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace ngraph;

namespace
{
    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value, testing::AssertionResult>::type
        compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                       const std::shared_ptr<ngraph::runtime::Tensor>& results,
                       const size_t tolerance_bits)
    {
        const auto expected = expected_results->get_vector<T>();
        const auto result = read_vector<T>(results);

        return ngraph::test::all_close_f(expected, result, tolerance_bits);
    }

    testing::AssertionResult
        compare_with_fp_tolerance(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                                  const std::shared_ptr<ngraph::runtime::Tensor>& results,
                                  const float tolerance)
    {
        auto comparison_result = testing::AssertionSuccess();

        const auto expected = expected_results->get_vector<float>();
        const auto result = read_vector<float>(results);

        Shape out_shape = expected_results->get_shape();

        size_t num_of_elems = shape_size(out_shape);
        std::stringstream msg;

        msg << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

        bool rc = true;

        for (std::size_t j = 0; j < num_of_elems; ++j)
        {
            float diff = std::abs(result[j] - expected[j]);
            if (diff > tolerance)
            {
                msg << expected[j] << " is not close to " << result[j] << " at index " << j << "\n";
                rc = false;
            }
        }

        if (!rc)
        {
            comparison_result = testing::AssertionFailure();
        }

        comparison_result << msg.str();
        return comparison_result;
    }

    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
        compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                       const std::shared_ptr<ngraph::runtime::Tensor>& results,
                       const size_t)
    {
        const auto expected = expected_results->get_vector<T>();
        const auto result = read_vector<T>(results);

        return ngraph::test::all_close(expected, result);
    }

    // used for float16 and bfloat 16 comparisons
    template <typename T>
    typename std::enable_if<std::is_class<T>::value, testing::AssertionResult>::type
        compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                       const std::shared_ptr<ngraph::runtime::Tensor>& results,
                       const size_t tolerance_bits)
    {
        const auto expected = expected_results->get_vector<T>();
        const auto result = read_vector<T>(results);

        // TODO: add testing infrastructure for float16 and bfloat16 to avoid cast to double
        std::vector<double> expected_double(expected.size());
        std::vector<double> result_double(result.size());

        NGRAPH_CHECK(expected.size() == result.size(),
                     "Number of expected and computed results don't match");

        for (int i = 0; i < expected.size(); ++i)
        {
            expected_double[i] = static_cast<double>(expected[i]);
            result_double[i] = static_cast<double>(result[i]);
        }

        return ngraph::test::all_close_f(expected_double, result_double, tolerance_bits);
    }
}; // namespace

test::INTERPRETER_Engine::INTERPRETER_Engine(const std::shared_ptr<Function> function)
    : m_function{function}
{
    m_backend = ngraph::runtime::Backend::create(NG_BACKEND_NAME, false); // static INT backend
    m_executable = m_backend->compile(m_function);
    for (auto i = 0; i < m_function->get_output_size(); ++i)
    {
        m_result_tensors.push_back(m_backend->create_tensor(m_function->get_output_element_type(i),
                                                            m_function->get_output_shape(i)));
    }
}

test::INTERPRETER_Engine::INTERPRETER_Engine(const std::shared_ptr<Function> function,
                                             INTERPRETER_Engine::DynamicBackendTag)
    : m_function{function}
{
    m_backend = ngraph::runtime::Backend::create(NG_BACKEND_NAME, true); // dynamic INT backend
    m_executable = m_backend->compile(m_function);
    for (auto i = 0; i < m_function->get_output_size(); ++i)
    {
        m_result_tensors.push_back(m_backend->create_dynamic_tensor(
            m_function->get_output_element_type(i), m_function->get_output_partial_shape(i)));
    }
}

test::INTERPRETER_Engine test::INTERPRETER_Engine::dynamic(const std::shared_ptr<Function> function)
{
    return INTERPRETER_Engine{function, DynamicBackendTag{}};
}

void test::INTERPRETER_Engine::infer()
{
    const auto& function_results = m_function->get_results();
    NGRAPH_CHECK(m_expected_outputs.size() == function_results.size(),
                 "Expected number of outputs is different from the function's number "
                 "of results.");
    m_executable->call_with_validate(m_result_tensors, m_input_tensors);
}

testing::AssertionResult
    test::INTERPRETER_Engine::compare_results_with_tolerance_as_fp(const float tolerance)
{
    auto comparison_result = testing::AssertionSuccess();

    for (size_t i = 0; i < m_expected_outputs.size(); ++i)
    {
        const auto& result_tensor = m_result_tensors.at(i);
        const auto& expected_result_constant = m_expected_outputs.at(i);
        const auto& element_type = result_tensor->get_element_type();

        const auto& expected_shape = expected_result_constant->get_shape();
        const auto& result_shape = result_tensor->get_shape();

        if (expected_shape != result_shape)
        {
            comparison_result = testing::AssertionFailure();
            comparison_result << "Computed data shape does not match the expected shape for output "
                              << i << std::endl;
            break;
        }

        switch (element_type)
        {
        case element::Type_t::f32:
            comparison_result =
                compare_with_fp_tolerance(expected_result_constant, result_tensor, tolerance);
            break;
        default:
            comparison_result = testing::AssertionFailure()
                                << "Unsupported data type encountered in "
                                   "'compare_results_with_tolerance_as_fp' method";
        }

        if (comparison_result == testing::AssertionFailure())
        {
            break;
        }
    }

    return comparison_result;
}

testing::AssertionResult test::INTERPRETER_Engine::compare_results(const size_t tolerance_bits)
{
    auto comparison_result = testing::AssertionSuccess();

    for (size_t i = 0; i < m_expected_outputs.size(); ++i)
    {
        const auto& result_tensor = m_result_tensors.at(i);
        const auto& expected_result_constant = m_expected_outputs.at(i);
        const auto& element_type = result_tensor->get_element_type();

        const auto& expected_shape = expected_result_constant->get_shape();
        const auto& result_shape = result_tensor->get_shape();

        if (expected_shape != result_shape)
        {
            comparison_result = testing::AssertionFailure();
            comparison_result << "Computed data shape does not match the expected shape for output "
                              << i << std::endl;
            break;
        }

        switch (element_type)
        {
        case element::Type_t::f16:
            comparison_result = compare_values<ngraph::float16>(
                expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::bf16:
            comparison_result = compare_values<ngraph::bfloat16>(
                expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f32:
            comparison_result =
                compare_values<float>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f64:
            comparison_result =
                compare_values<double>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i8:
            comparison_result =
                compare_values<int8_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i16:
            comparison_result =
                compare_values<int16_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i32:
            comparison_result =
                compare_values<int32_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i64:
            comparison_result =
                compare_values<int64_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u8:
            comparison_result =
                compare_values<uint8_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u16:
            comparison_result =
                compare_values<uint16_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u32:
            comparison_result =
                compare_values<uint32_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u64:
            comparison_result =
                compare_values<uint64_t>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        case element::Type_t::boolean:
            comparison_result =
                compare_values<char>(expected_result_constant, result_tensor, tolerance_bits);
            break;
        default:
            comparison_result = testing::AssertionFailure()
                                << "Unsupported data type encountered in 'compare_results' method";
        }

        if (comparison_result == testing::AssertionFailure())
        {
            break;
        }
    }

    return comparison_result;
}

void test::INTERPRETER_Engine::reset()
{
    m_input_index = 0;
    m_output_index = 0;
    m_expected_outputs.clear();
    m_input_tensors.clear();
}
