// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    using UnaryFuncCreator =
        std::function<std::shared_ptr<Function>(const ngraph::element::Type&, const PartialShape&)>;

    template <typename OpType, typename... Args>
    typename std::enable_if<std::is_base_of<op::Op, OpType>::value, UnaryFuncCreator>::type
        unary_func(Args&&... args)
    {
        return [args...](const ngraph::element::Type& ele_type, const PartialShape& pshape) {
            auto X = make_shared<op::Parameter>(ele_type, pshape);
            auto Y = make_shared<OpType>(X, args...);
            return make_shared<Function>(Y, ParameterVector{X});
        };
    }

    struct Tolerance
    {
        const int tolerance_bits;
        const float tolerance_fp;
        bool using_bits;
        Tolerance()
            : tolerance_bits(-1)
            , tolerance_fp(0)
            , using_bits(true)
        {
        }
        Tolerance(const int tolerance_bits)
            : tolerance_bits(tolerance_bits)
            , tolerance_fp(0)
            , using_bits(true)
        {
        }
        Tolerance(const float tolerance_fp)
            : tolerance_bits(-1)
            , tolerance_fp(tolerance_fp)
            , using_bits(false)
        {
        }
        Tolerance(const double tolerance_fp)
            : tolerance_bits(-1)
            , tolerance_fp(tolerance_fp)
            , using_bits(false)
        {
        }
    };

    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& test_input,
                    const std::vector<value_type>& test_expected,
                    PartialShape dynamic_shape,
                    Shape static_shape,
                    Tolerance tol = {})
    {
        // bool must_support_dynamic = dynamic_shape.is_dynamic();

        // expand or shrink the size of input & expected values
        auto static_size = shape_size(static_shape);
        std::vector<value_type> input;
        std::vector<value_type> expected;
        for (size_t i = 0; i < static_size; i++)
        {
            input.push_back(test_input[i % test_input.size()]);
            expected.push_back(test_expected[i % test_expected.size()]);
        }

        auto test_case = test::TestCase<TestEngine>(f);
        test_case.template add_input<value_type>(input);
        test_case.template add_expected_output<value_type>(static_shape, expected);
        if (tol.using_bits)
        {
            if (tol.tolerance_bits >= 0)
                test_case.run(tol.tolerance_bits);
            else
                test_case.run();
        }
        else
        {
            test_case.run_with_tolerance_as_fp(tol.tolerance_fp);
        }
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    PartialShape pshape,
                    Shape sshape,
                    Tolerance tol = {})
    {
        test_unary<TestEngine, element_type>(
            creator(element_type, pshape), input, expected, pshape, sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    PartialShape pshape,
                    Shape sshape,
                    Tolerance tol = {})
    {
        std::vector<value_type> expected;
        // in case input cannot be precisely expressed by value_type
        // converting them first to value_type then back to double
        // would generate result more close to expectation
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<TestEngine, element_type>(
            creator(element_type, pshape), input, expected, pshape, sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    Tolerance tol = {})
    {
        Shape sshape({input.size()});
        test_unary<TestEngine, element_type, Creator>(
            creator, input, expected, PartialShape(sshape), sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    Tolerance tol = {})
    {
        Shape sshape({input.size()});
        test_unary<TestEngine, element_type, Creator>(
            creator, input, func, PartialShape(sshape), sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    PartialShape pshape,
                    Shape sshape,
                    Tolerance tol = {})
    {
        std::vector<value_type> expected;
        // in case input cannot be precisely expressed by value_type
        // converting them first to value_type then back to double
        // would generate result more close to expectation
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<TestEngine, element_type>(f, input, expected, pshape, sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    Tolerance tol = {})
    {
        Shape sshape({input.size()});
        test_unary<TestEngine, element_type>(f, input, expected, PartialShape(sshape), sshape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    Tolerance tol = {})
    {
        Shape sshape({input.size()});
        test_unary<TestEngine, element_type>(f, input, func, PartialShape(sshape), sshape, tol);
    }
}
