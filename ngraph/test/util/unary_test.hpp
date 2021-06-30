// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

namespace ngraph
{
    namespace test
    {
        template <typename V>
        struct ArgumentTolerance
        {
            ArgumentTolerance() = default;
            ArgumentTolerance(V v)
                : in_use{true}
                , value{v}
            {
            }
            bool in_use{false};
            V value{};
        };

        using BitTolerance = ArgumentTolerance<int>;
        using FloatTolerance = ArgumentTolerance<float>;

        struct Tolerance
        {
            BitTolerance bit{};
            FloatTolerance fp{};

            Tolerance() = default;
            Tolerance(const float tolerance)
                : fp(tolerance)
            {
            }
            Tolerance(const int tolerance)
                : bit(tolerance)
            {
            }
        };

        template <typename TestEngine,
                  ngraph::element::Type_t et_input,
                  ngraph::element::Type_t et_output>
        class unary_test;

        struct rand_normal_tag
        {
        };
        struct rand_uniform_int_tag
        {
        };
        struct rand_uniform_real_tag
        {
        };

        template <ngraph::element::Type_t eleType>
        class Data
        {
        public:
            using T = ngraph::fundamental_type_for<eleType>;

            // random (normal-distrubition) generate data
            Data(rand_normal_tag, double mean, double stddev, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::normal_distribution<> d{mean, stddev};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            // random (uniform-int-distrubition) generate data
            Data(rand_uniform_int_tag, int64_t low, int64_t high, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::uniform_int_distribution<int64_t> d{low, high};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            // random (uniform-real-distrubition) generate data
            Data(rand_uniform_real_tag, double low, double high, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape(s)
            {
                std::mt19937 gen{0};
                std::uniform_real_distribution<> d{low, high};
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = d(gen);
            }

            Data() = default;

            // only vector, no shape specified
            Data(const std::vector<T>& v)
                : value(v)
                , shape{}
                , no_shape(true)
            {
            }

            // tensor with shape
            Data(const std::vector<T>& v, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape{s}
            {
                for (size_t i = 0; i < value.size(); ++i)
                    value[i] = v[i % v.size()];
            }

            // caller have to pass two level of braces when aggregate-initialize input argument of
            // type std::vector<T>. by adding an overload of type std::initializer_list<T>, we can
            // save one level of braces
            Data(const std::initializer_list<T>& v)
                : value(v.size())
                , shape{}
                , no_shape(true)
            {
                std::copy(v.begin(), v.end(), value.begin());
            }

            // we would duplicate by circular-copy or truncate initializer list when number of
            // element provided in initializer list is not matching the desired shape
            Data(const std::initializer_list<T>& v, const ngraph::Shape& s)
                : value(ngraph::shape_size(s))
                , shape{s}
            {
                auto it = v.begin();
                for (size_t i = 0; i < value.size(); ++i, ++it)
                {
                    if (it == v.end())
                        it = v.begin();
                    value[i] = *it;
                }
            }

        private:
            std::vector<T> value{};
            ngraph::Shape shape{};
            bool no_shape = false;

            template <typename TestEngine,
                      ngraph::element::Type_t et_input,
                      ngraph::element::Type_t et_output>
            friend class unary_test;
        };

        template <typename TestEngine,
                  ngraph::element::Type_t et_input,
                  ngraph::element::Type_t et_output = et_input>
        class unary_test
        {
        public:
            unary_test(std::shared_ptr<ngraph::Function> function)
                : m_function(function)
            {
            }

            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output>
            void test(const std::initializer_list<Data<et_i>>& inputs,
                      const std::initializer_list<Data<et_o>>& expeted,
                      Tolerance tol = {})
            {
                if (m_function->is_dynamic())
                {
                    auto test_case =
                        ngraph::test::TestCase<TestEngine, ngraph::test::TestCaseType::DYNAMIC>(
                            m_function);
                    do_test<et_i, et_o>(test_case, inputs, expeted, tol);
                }
                else
                {
                    auto test_case =
                        ngraph::test::TestCase<TestEngine, ngraph::test::TestCaseType::STATIC>(
                            m_function);
                    do_test<et_i, et_o>(test_case, inputs, expeted, tol);
                }
            }

            // SISO: single inputs, single output
            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output>
            void test(const Data<et_i>& input, const Data<et_o>& expeted, Tolerance tol = {})
            {
                test<et_i, et_o>(std::initializer_list<Data<et_i>>{input},
                                 std::initializer_list<Data<et_o>>{expeted},
                                 tol);
            }

            // MISO: multiple inputs, single output
            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output>
            void test(const std::initializer_list<Data<et_i>>& inputs,
                      const Data<et_o>& expeted,
                      Tolerance tol = {})
            {
                test<et_i, et_o>(inputs, std::initializer_list<Data<et_o>>{expeted}, tol);
            }

            // this overload supports passing a predictor with overloaded i/o types
            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output,
                      typename Ti = ngraph::fundamental_type_for<et_i>,
                      typename To = ngraph::fundamental_type_for<et_o>>
            void test(const std::initializer_list<Data<et_i>>& inputs,
                      To (*elewise_predictor)(Ti),
                      Tolerance tol = {})
            {
                auto first = inputs.begin();
                Data<et_o> expeted;
                expeted.shape = first->shape;
                expeted.no_shape = first->no_shape;
                expeted.value.resize(first->value.size());

                for (size_t i = 0; i < first->value.size(); i++)
                    expeted.value[i] = elewise_predictor(first->value[i]);

                test<et_i, et_o>(inputs, expeted, tol);
            }

            // this overload supports passing a lambda as predictor
            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output,
                      typename Predictor>
            void test(const std::initializer_list<Data<et_i>>& inputs,
                      Predictor elewise_predictor,
                      Tolerance tol = {})
            {
                auto first = inputs.begin();
                Data<et_o> expeted;
                expeted.shape = first->shape;
                expeted.no_shape = first->no_shape;
                expeted.value.resize(first->value.size());

                for (size_t i = 0; i < first->value.size(); i++)
                    expeted.value[i] = elewise_predictor(first->value[i]);

                test<et_i, et_o>(inputs, expeted, tol);
            }

            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output,
                      typename Ti = ngraph::fundamental_type_for<et_i>,
                      typename To = ngraph::fundamental_type_for<et_o>>
            void test(const Data<et_i>& input, To (*elewise_predictor)(Ti), Tolerance tol = {})
            {
                test<et_i, et_o>(std::initializer_list<Data<et_i>>{input}, elewise_predictor, tol);
            }

            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output,
                      typename Predictor>
            void test(const Data<et_i>& input, Predictor elewise_predictor, Tolerance tol = {})
            {
                test<et_i, et_o>(std::initializer_list<Data<et_i>>{input}, elewise_predictor, tol);
            }

        private:
            std::shared_ptr<ngraph::Function> m_function;

            template <ngraph::element::Type_t et_i = et_input,
                      ngraph::element::Type_t et_o = et_output,
                      typename TC,
                      typename Ti = ngraph::fundamental_type_for<et_i>,
                      typename To = ngraph::fundamental_type_for<et_o>>
            void do_test(TC& test_case,
                         const std::initializer_list<Data<et_i>>& input,
                         const std::initializer_list<Data<et_o>>& expeted,
                         Tolerance tol)
            {
                for (const auto& item : input)
                {
                    if (item.no_shape)
                        test_case.template add_input<Ti>(item.value);
                    else
                        test_case.template add_input<Ti>(item.shape, item.value);
                }

                for (const auto& item : expeted)
                {
                    if (item.no_shape)
                        test_case.template add_expected_output<To>(item.value);
                    else
                        test_case.template add_expected_output<To>(item.shape, item.value);
                }

                if (tol.bit.in_use)
                    test_case.run(tol.bit.value);
                else if (tol.fp.in_use)
                    test_case.run_with_tolerance_as_fp(tol.fp.value);
                else
                    test_case.run();
            }
        };

        template <typename TestEngine,
                  typename OpType,
                  ngraph::element::Type_t et,
                  typename... Args>
        unary_test<TestEngine, et, et>
            make_unary_test(const ngraph::PartialShape& pshape = ngraph::PartialShape::dynamic(),
                            Args&&... args)
        {
            auto param = std::make_shared<ngraph::op::Parameter>(et, pshape);
            auto op = std::make_shared<OpType>(param, std::forward<Args>(args)...);
            auto function = std::make_shared<ngraph::Function>(op, ngraph::ParameterVector{param});
            return unary_test<TestEngine, et, et>(function);
        }
    }
}