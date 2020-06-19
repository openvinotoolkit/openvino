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

using namespace ngraph;

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

testing::AssertionResult test::INTERPRETER_Engine::compare_results(const size_t tolerance_bits)
{
    m_tolerance_bits = tolerance_bits;

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

        const auto values_match = m_value_comparators.at(element_type);

        comparison_result = values_match(expected_result_constant, result_tensor);

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
