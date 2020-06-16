#pragma once
#include "../../util/all_close.hpp"
#include "../../util/all_close_f.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph
{
    namespace test
    {
        // TODO - implement when IE_CPU engine is done
        class INTERPRETER_Engine
        {
        public:
            INTERPRETER_Engine(const std::shared_ptr<Function> function);

            void infer()
            {
                const auto& function_results = m_function->get_results();
                NGRAPH_CHECK(m_expected_outputs.size() == function_results.size(),
                             "Expected number of outputs is different from the function's number "
                             "of results.");
                m_executable->call_with_validate(m_result_tensors, m_input_tensors);
            }

            testing::AssertionResult
                compare_results(const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS);

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
            const std::shared_ptr<Function> m_function;
            std::shared_ptr<runtime::Backend> m_backend;
            std::shared_ptr<ngraph::runtime::Executable> m_executable;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_input_tensors;
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> m_result_tensors;
            std::vector<std::shared_ptr<ngraph::op::Constant>> m_expected_outputs;
            size_t m_input_index = 0;
            size_t m_output_index = 0;
            size_t m_tolerance_bits = DEFAULT_DOUBLE_TOLERANCE_BITS;

            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value,
                                    testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                return ngraph::test::all_close_f(expected, result, m_tolerance_bits);
            }

            template <typename T>
            typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
                compare_values(const std::shared_ptr<ngraph::op::Constant>& expected_results,
                               const std::shared_ptr<ngraph::runtime::Tensor>& results)
            {
                const auto expected = expected_results->get_vector<T>();
                const auto result = read_vector<T>(results);

                return ngraph::test::all_close(expected, result);
            }

            // used for float16 and bfloat 16 comparisons
            template <typename T>
            typename std::enable_if<std::is_class<T>::value, testing::AssertionResult>::type
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

                return ngraph::test::all_close_f(expected_double, result_double, m_tolerance_bits);
            }

            using value_comparator_function = std::function<testing::AssertionResult(
                const std::shared_ptr<ngraph::op::Constant>&,
                const std::shared_ptr<ngraph::runtime::Tensor>&)>;

#define REGISTER_COMPARATOR(element_type_, type_)                                                  \
    {                                                                                              \
        ngraph::element::Type_t::element_type_,                                                    \
            std::bind(&INTERPRETER_Engine::compare_values<type_>,                                  \
                      this,                                                                        \
                      std::placeholders::_1,                                                       \
                      std::placeholders::_2)                                                       \
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
        };
    }
}