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

#include "test_case.hpp"

#include "gtest/gtest.h"
#include "ngraph/assertion.hpp"

ngraph::test::NgraphTestCase::NgraphTestCase(const std::shared_ptr<Function>& function,
                                             const std::string& backend_name,
                                             const BackendMode mode)
    : m_function(function)
{
    if (mode == BackendMode::STATIC)
    {
        NGRAPH_CHECK(!m_function->is_dynamic(),
                     "For dynamic function using dynamic backend is expected.");
    }

    // IE backend test should not be run with dynamic backend wrapper
    const bool use_dynamic =
        mode == BackendMode::DYNAMIC && backend_name.find("IE") == std::string::npos;

    m_backend = ngraph::runtime::Backend::create(backend_name, use_dynamic);
    m_executable = m_backend->compile(m_function);
    for (auto i = 0; i < m_function->get_output_size(); ++i)
    {
        const auto& output_tensor =
            (use_dynamic)
                ? m_backend->create_dynamic_tensor(m_function->get_output_element_type(i),
                                                   m_function->get_output_partial_shape(i))
                : m_backend->create_tensor(m_function->get_output_element_type(i),
                                           m_function->get_output_shape(i));

        m_result_tensors.emplace_back(output_tensor);
    }
}

void ngraph::test::NgraphTestCase::run(size_t tolerance_bits)
{
    m_tolerance_bits = tolerance_bits;
    const auto& function_results = m_function->get_results();
    NGRAPH_CHECK(m_expected_outputs.size() == function_results.size(),
                 "Expected number of outputs is different from the function's number of results.");
    m_executable->call_with_validate(m_result_tensors, m_input_tensors);

    for (size_t i = 0; i < m_expected_outputs.size(); ++i)
    {
        const auto& result_tensor = m_result_tensors.at(i);
        const auto& expected_result_constant = m_expected_outputs.at(i);
        const auto& element_type = result_tensor->get_element_type();

        EXPECT_EQ(expected_result_constant->get_output_size(), 1);
        const auto& expected_shape = expected_result_constant->get_shape();
        const auto& result_shape = result_tensor->get_shape();

        EXPECT_EQ(expected_shape, result_shape);

        if (m_value_comparators.count(element_type) == 0)
        {
            NGRAPH_FAIL() << "Please add support for " << element_type
                          << " to ngraph::test::NgraphTestCase::run()";
        }
        else
        {
            auto values_match = m_value_comparators.at(element_type);

            EXPECT_TRUE(values_match(expected_result_constant, result_tensor));
        }
    }
    m_input_index = 0;
    m_output_index = 0;
    m_expected_outputs.clear();
    m_input_tensors.clear();
}

ngraph::test::NgraphTestCase& ngraph::test::NgraphTestCase::dump_results(bool dump)
{
    m_dump_results = dump;
    return *this;
}
