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

#pragma once

#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/engine_traits.hpp"
#include "util/engine/test_case_engine.hpp"

namespace ngraph
{
    namespace test
    {
        class INTERPRETER_Engine : public TestCaseEngine
        {
        public:
            INTERPRETER_Engine(const std::shared_ptr<Function> function);

            static INTERPRETER_Engine dynamic(const std::shared_ptr<Function> function);

            void infer() override;

            testing::AssertionResult compare_results(
                const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS) override;

            testing::AssertionResult
                compare_results_with_tolerance_as_fp(const float tolerance = 1.0e-5f) override;

            void reset() override;

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                const auto params = m_function->get_parameters();
                auto tensor =
                    m_backend->create_tensor(params.at(m_input_index)->get_element_type(), shape);

                copy_data(tensor, values);

                m_input_tensors.push_back(tensor);

                ++m_input_index;
            }

            template <typename T>
            void add_expected_output(const ngraph::Shape& expected_shape,
                                     const std::vector<T>& values)
            {
                const auto results = m_function->get_results();

                const auto function_output_type = results.at(m_output_index)->get_element_type();

                m_expected_outputs.emplace_back(std::make_shared<ngraph::op::Constant>(
                    function_output_type, expected_shape, values));

                ++m_output_index;
            }

        private:
            struct DynamicBackendTag
            {
            };
            /// A private constructor that should only be used from the dynamic() member function
            INTERPRETER_Engine(const std::shared_ptr<Function> function, DynamicBackendTag);

            static constexpr const char* NG_BACKEND_NAME = "INTERPRETER";

            const std::shared_ptr<Function> m_function;
            std::shared_ptr<runtime::Backend> m_backend;
            std::shared_ptr<ngraph::runtime::Executable> m_executable;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_input_tensors;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_result_tensors;
            std::vector<std::shared_ptr<ngraph::op::Constant>> m_expected_outputs;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
        };

        template <>
        struct supports_dynamic<INTERPRETER_Engine>
        {
            static constexpr bool value = true;
        };
    }
}
