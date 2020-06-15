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

#include <utility>

#include "all_close.hpp"
#include "all_close_f.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        /// \brief Indicates which version of backend (dynamic or static) should be using in
        /// NgraphTestCase
        enum class BackendMode
        {
            // Use static version of backend
            STATIC,
            // Use dynamic version of backend
            DYNAMIC
        };

        class NgraphTestCase
        {
        public:
            NgraphTestCase(const std::shared_ptr<Function>& function,
                           const std::string& backend_name,
                           BackendMode mode = BackendMode::STATIC);

            NgraphTestCase(const std::shared_ptr<Function>& function);

            /// \brief Makes the test case print the expected and computed values to the console.
            ///        This should only be used for debugging purposes.
            ///
            /// Just before the assertion is done, the current test case will gather expected and
            /// computed values, format them as 2 columns and print out to the console along with
            /// a corresponding index in the vector.
            ///
            /// \param dump - Indicates if the test case should perform the console printout
            NgraphTestCase& dump_results(bool dump = true);

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                const auto params = m_function->get_parameters();
                NGRAPH_CHECK(m_input_index < params.size(),
                             "All function parameters already have inputs.");

                const auto& input_pshape = params.at(m_input_index)->get_partial_shape();
                NGRAPH_CHECK(input_pshape.compatible(shape),
                             "Passed input shape is not compatible with nGraph function.");

                auto tensor =
                    m_backend->create_tensor(params.at(m_input_index)->get_element_type(), shape);
                copy_data(tensor, values);

                m_input_tensors.push_back(tensor);

                ++m_input_index;
            }

            template <typename T>
            void add_input(const std::vector<T>& values)
            {
                const auto& input_pshape =
                    m_function->get_parameters().at(m_input_index)->get_partial_shape();
                NGRAPH_CHECK(input_pshape.is_static(),
                             "Input data shape must be provided, if shape defined in Functions is "
                             "not fully known.");

                add_input<T>(input_pshape.to_shape(), values);
            }

            template <typename T>
            void add_input_from_file(const Shape& shape,
                                     const std::string& basepath,
                                     const std::string& filename)
            {
                auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_input_from_file<T>(shape, filepath);
            }

            template <typename T>
            void add_input_from_file(const std::string& basepath, const std::string& filename)
            {
                auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_input_from_file<T>(filepath);
            }

            template <typename T>
            void add_input_from_file(const Shape& shape, const std::string& filepath)
            {
                auto value = read_binary_file<T>(filepath);
                add_input<T>(shape, value);
            }

            template <typename T>
            void add_input_from_file(const std::string& filepath)
            {
                auto value = read_binary_file<T>(filepath);
                add_input<T>(value);
            }

            template <typename T>
            void add_multiple_inputs(const std::vector<Shape> shapes,
                                     const std::vector<std::vector<T>>& vector_of_values)
            {
                NGRAPH_CHECK(shapes.size() == vector_of_values.size(),
                             "Size of shapes and vector_of_values vectors must be the same.");

                for (auto i = 0; i < vector_of_values.size(); ++i)
                {
                    add_input<T>(shapes[i], vector_of_values[i]);
                }
            }

            template <typename T>
            void add_multiple_inputs(const std::vector<std::vector<T>>& vector_of_values)
            {
                for (const auto& value : vector_of_values)
                {
                    add_input<T>(value);
                }
            }

            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
                auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                auto function_output_type = results.at(m_output_index)->get_element_type();

                const auto& output_pshape = results.at(m_output_index)->get_output_partial_shape(0);
                NGRAPH_CHECK(
                    output_pshape.compatible(expected_shape),
                    "nGraph function generated an unexpected output shape. Expected shape: ",
                    expected_shape,
                    " Output shape: ",
                    output_pshape);

                m_expected_outputs.emplace_back(std::make_shared<ngraph::op::Constant>(
                    function_output_type, expected_shape, values));

                ++m_output_index;
            }

            template <typename T>
            void add_expected_output(const std::vector<T>& values)
            {
                auto shape = m_function->get_results().at(m_output_index)->get_shape();
                add_expected_output<T>(shape, values);
            }

            template <typename T>
            void add_expected_output_from_file(ngraph::Shape expected_shape,
                                               const std::string& basepath,
                                               const std::string& filename)
            {
                auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_expected_output_from_file<T>(expected_shape, filepath);
            }

            template <typename T>
            void add_expected_output_from_file(ngraph::Shape expected_shape,
                                               const std::string& filepath)
            {
                auto value = read_binary_file<T>(filepath);
                add_expected_output<T>(expected_shape, value);
            }

            virtual ::testing::AssertionResult
                run(size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS);

        private:
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value,
                                    ::testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                if (m_dump_results)
                {
                    std::cout << get_results_str<T>(expected, result, expected.size());
                }

                return ngraph::test::all_close_f(expected, result, m_tolerance_bits);
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                if (m_dump_results)
                {
                    std::cout << get_results_str<T>(expected, result, expected.size());
                }

                return ngraph::test::all_close(expected, result);
            }

            // used for float16 and bfloat 16 comparisons
            template <typename T>
            typename std::enable_if<std::is_class<T>::value, ::testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                // TODO: add testing infrastructure for float16 and bfloat16 to avoid cast to double
                std::vector<double> expected_double(expected.size());
                std::vector<double> result_double(result.size());
                assert(expected.size() == result.size() && "expected and result size must match");
                for (int i = 0; i < expected.size(); ++i)
                {
                    expected_double[i] = static_cast<double>(expected[i]);
                    result_double[i] = static_cast<double>(result[i]);
                }

                if (m_dump_results)
                {
                    std::cout << get_results_str<double>(
                        expected_double, result_double, expected.size());
                }

                return ngraph::test::all_close_f(expected_double, result_double, m_tolerance_bits);
            }

            using value_comparator_function = std::function<::testing::AssertionResult(
                const std::shared_ptr<ngraph::op::Constant>&,
                const std::shared_ptr<ngraph::runtime::Tensor>&)>;

#define REGISTER_COMPARATOR(element_type_, type_)                                                  \
    {                                                                                              \
        ngraph::element::Type_t::element_type_, std::bind(&NgraphTestCase::compare_values<type_>,  \
                                                          this,                                    \
                                                          std::placeholders::_1,                   \
                                                          std::placeholders::_2)                   \
    }

            std::map<ngraph::element::Type_t, value_comparator_function> m_value_comparators = {
                REGISTER_COMPARATOR(f16, ngraph::float16),
                REGISTER_COMPARATOR(bf16, ngraph::bfloat16),
                REGISTER_COMPARATOR(f32, float),
                REGISTER_COMPARATOR(f64, double),
                REGISTER_COMPARATOR(i8, int8_t),
                REGISTER_COMPARATOR(i16, int16_t),
                REGISTER_COMPARATOR(i32, int32_t),
                REGISTER_COMPARATOR(i64, int64_t),
                REGISTER_COMPARATOR(u8, uint8_t),
                REGISTER_COMPARATOR(u16, uint16_t),
                REGISTER_COMPARATOR(u32, uint32_t),
                REGISTER_COMPARATOR(u64, uint64_t),
                REGISTER_COMPARATOR(boolean, char),
            };
#undef REGISTER_COMPARATOR

        protected:
            std::shared_ptr<Function> m_function;
            std::shared_ptr<runtime::Backend> m_backend;
            std::shared_ptr<ngraph::runtime::Executable> m_executable;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_input_tensors;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_result_tensors;
            std::vector<std::shared_ptr<ngraph::op::Constant>> m_expected_outputs;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
            bool m_dump_results = false;
            int m_tolerance_bits = DEFAULT_DOUBLE_TOLERANCE_BITS;
        };

        template <typename Engine>
        class TestCase
        {
        public:
            TestCase(const std::shared_ptr<Function>& function)
                : m_function{function}
                , m_engine{function}
            {
            }

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                const auto params = m_function->get_parameters();
                NGRAPH_CHECK(m_input_index < params.size(),
                             "All function parameters already have inputs.");

                const auto& input_pshape = params.at(m_input_index)->get_partial_shape();
                NGRAPH_CHECK(input_pshape.compatible(shape),
                             "Passed input shape is not compatible with nGraph function.");

                m_engine.add_input<T>(shape, values);

                ++m_input_index;
            }

            template <typename T>
            void add_input(const std::vector<T>& values)
            {
                const auto& input_pshape =
                    m_function->get_parameters().at(m_input_index)->get_partial_shape();
                NGRAPH_CHECK(input_pshape.is_static(),
                             "Input data shape must be provided, if shape defined in Functions is "
                             "not fully known.");

                add_input<T>(input_pshape.to_shape(), values);
            }

            template <typename T>
            void add_expected_output(const Shape& expected_shape, const std::vector<T>& values)
            {
                auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                auto function_output_type = results.at(m_output_index)->get_element_type();

                const auto& output_pshape = results.at(m_output_index)->get_output_partial_shape(0);
                NGRAPH_CHECK(
                    output_pshape.compatible(expected_shape),
                    "nGraph function generated an unexpected output shape. Expected shape: ",
                    expected_shape,
                    " Output shape: ",
                    output_pshape);

                m_engine.add_expected_output<T>(expected_shape, values);

                ++m_output_index;
            }

            ::testing::AssertionResult
                run(size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS)
            {
                std::cout << "Running TestCase\n";
                m_engine.infer();
                const auto res = m_engine.compare_results();
                EXPECT_EQ(res, testing::AssertionSuccess());
                // const auto results = m_engine.template output_data<float>();
                // EXPECT_TRUE(results.size() > 0);
                // return ::testing::AssertionSuccess();
            }

        private:
            Engine m_engine;
            std::shared_ptr<Function> m_function;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
        };
    }
}
