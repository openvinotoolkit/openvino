// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "all_close.hpp"
#include "all_close_f.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "test_tools.hpp"
#include "util/engine/engine_factory.hpp"

namespace ngraph
{
    namespace test
    {
        std::shared_ptr<Function> function_from_ir(const std::string& xml_path,
                                                   const std::string& bin_path = {});

        template <typename Engine, TestCaseType tct = TestCaseType::STATIC>
        class TestCase
        {
        public:
            TestCase(const std::shared_ptr<Function>& function)
                : m_engine{create_engine<Engine>(function, tct)}
                , m_function{function}
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
                             "Provided input shape ",
                             shape,
                             " is not compatible with nGraph function's expected input shape ",
                             input_pshape,
                             " for input ",
                             m_input_index);

                m_engine.template add_input<T>(shape, values);

                ++m_input_index;
            }

            template <typename T>
            void add_input(const std::vector<T>& values)
            {
                const auto& input_pshape =
                    m_function->get_parameters().at(m_input_index)->get_partial_shape();

                NGRAPH_CHECK(input_pshape.is_static(),
                             "Input number ",
                             m_input_index,
                             " in the tested graph has dynamic shape. You need to provide ",
                             "shape information when setting values for this input.");

                add_input<T>(input_pshape.to_shape(), values);
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
            void add_input_from_file(const Shape& shape,
                                     const std::string& basepath,
                                     const std::string& filename)
            {
                const auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_input_from_file<T>(shape, filepath);
            }

            template <typename T>
            void add_input_from_file(const std::string& basepath, const std::string& filename)
            {
                const auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_input_from_file<T>(filepath);
            }

            template <typename T>
            void add_input_from_file(const Shape& shape, const std::string& filepath)
            {
                const auto value = read_binary_file<T>(filepath);
                add_input<T>(shape, value);
            }

            template <typename T>
            void add_input_from_file(const std::string& filepath)
            {
                const auto value = read_binary_file<T>(filepath);
                add_input<T>(value);
            }

            template <typename T>
            void add_expected_output(const Shape& expected_shape, const std::vector<T>& values)
            {
                const auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                const auto& output_pshape = results.at(m_output_index)->get_output_partial_shape(0);
                NGRAPH_CHECK(output_pshape.compatible(expected_shape),
                             "Provided expected output shape ",
                             expected_shape,
                             " is not compatible with nGraph function's output shape ",
                             output_pshape,
                             " for output ",
                             m_output_index);

                m_engine.template add_expected_output<T>(expected_shape, values);

                ++m_output_index;
            }

            template <typename T>
            void add_expected_output(const std::vector<T>& values)
            {
                const auto results = m_function->get_results();

                NGRAPH_CHECK(m_output_index < results.size(),
                             "All function results already have expected outputs.");

                const auto shape = results.at(m_output_index)->get_shape();
                add_expected_output<T>(shape, values);
            }

            template <typename T>
            void add_expected_output_from_file(const ngraph::Shape& expected_shape,
                                               const std::string& basepath,
                                               const std::string& filename)
            {
                const auto filepath = ngraph::file_util::path_join(basepath, filename);
                add_expected_output_from_file<T>(expected_shape, filepath);
            }

            template <typename T>
            void add_expected_output_from_file(const ngraph::Shape& expected_shape,
                                               const std::string& filepath)
            {
                const auto values = read_binary_file<T>(filepath);
                add_expected_output<T>(expected_shape, values);
            }

            void run(const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS)
            {
                m_engine.infer();
                const auto res = m_engine.compare_results(tolerance_bits);

                if (res != testing::AssertionSuccess())
                {
                    std::cout << res.message() << std::endl;
                }

                m_input_index = 0;
                m_output_index = 0;
                m_engine.reset();

                EXPECT_TRUE(res);
            }

            void run_with_tolerance_as_fp(const float tolerance = 1.0e-5f)
            {
                m_engine.infer();
                const auto res = m_engine.compare_results_with_tolerance_as_fp(tolerance);

                if (res != testing::AssertionSuccess())
                {
                    std::cout << res.message() << std::endl;
                }

                m_input_index = 0;
                m_output_index = 0;
                m_engine.reset();

                EXPECT_TRUE(res);
            }

        private:
            Engine m_engine;
            std::shared_ptr<Function> m_function;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
        };
    }
}
