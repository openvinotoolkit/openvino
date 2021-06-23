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
    // Tolerance can be specified in two forms (number of bits in mantissa or float-point
    // threshold), corresponding to two different comparing methods implemented in TestCase
    // classes. Following class acts as an encapsulation on these two options without introducing
    // too many overloadings on the function requiring such argument.
    struct ArgTolerance
    {
        const int tolerance_bits;
        const float tolerance_fp;
        bool using_bits;
        ArgTolerance()
            : tolerance_bits(-1)
            , tolerance_fp(0)
            , using_bits(true)
        {
        }
        ArgTolerance(const int tolerance_bits)
            : tolerance_bits(tolerance_bits)
            , tolerance_fp(0)
            , using_bits(true)
        {
        }
        ArgTolerance(const float tolerance_fp)
            : tolerance_bits(-1)
            , tolerance_fp(tolerance_fp)
            , using_bits(false)
        {
        }
        ArgTolerance(const double tolerance_fp)
            : tolerance_bits(-1)
            , tolerance_fp(tolerance_fp)
            , using_bits(false)
        {
        }
    };

    // ArgShape is used implicitly by caller of test_unary() in following ways:
    //
    //  {PartialShape{2,Dimension::dynamic(),3}, Shape{2, 4, 3}}
    //        A full specified form with both partial shape for ngraph function
    //        generation and a static shape for final input tensor.
    //
    //  Shape{2, 4, 3}
    //        A pure static shape for both ngraph function generation and input tensor
    //
    //  {}
    //        A unspecified shape, test_unary() will derive the actual shape from the size
    //        of the input vector as 1D shape.
    struct ArgShape
    {
        PartialShape dynamic_shape;
        Shape static_shape;
        bool initialzed;
        ArgShape(const PartialShape& dynamic_shape, const Shape& static_shape)
            : dynamic_shape(dynamic_shape)
            , static_shape(static_shape)
            , initialzed(true)
        {
        }
        ArgShape(const Shape& static_shape)
            : dynamic_shape(static_shape)
            , static_shape(static_shape)
            , initialzed(true)
        {
        }
        ArgShape()
            : dynamic_shape{}
            , static_shape{}
            , initialzed(false)
        {
        }
    };

    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& test_input,
                    const std::vector<value_type>& test_expected,
                    ArgShape ashape = {},
                    ArgTolerance tol = {})
    {
        if (!ashape.initialzed)
        {
            ashape = ArgShape(Shape({test_input.size()}));
        }
        // the size specified in static_shape may be bigger or smaller than the
        // number of data contained in test_input & test_expected, here we
        // sample test data repetitively to generate actual test data of required size.
        auto static_size = shape_size(ashape.static_shape);
        std::vector<value_type> input;
        std::vector<value_type> expected;
        for (size_t i = 0; i < static_size; i++)
        {
            input.push_back(test_input[i % test_input.size()]);
            expected.push_back(test_expected[i % test_expected.size()]);
        }

        auto test_case = test::TestCase<TestEngine>(f);
        test_case.template add_input<value_type>(input);
        test_case.template add_expected_output<value_type>(ashape.static_shape, expected);
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

    // overloading support passing reference function instead of expected results
    template <typename TestEngine,
              element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    ArgShape ashape = {},
                    ArgTolerance tol = {})
    {
        std::vector<value_type> expected;
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<TestEngine, element_type>(f, input, expected, ashape, tol);
    }

    // overloading support passing creator instead of ngraph function
    // Creator is any callable of following prototype:
    //
    //          std::shared_ptr<Function> (const ngraph::element::Type&, const PartialShape&)
    //
    // unary_func() is a helper function to generate a lambda as callable on simple unary ops with
    // optional arguments captured in the that lambda.
    //
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

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    ArgShape ashape = {},
                    ArgTolerance tol = {})
    {
        if (!ashape.initialzed)
        {
            ashape = ArgShape(Shape({input.size()}));
        }
        test_unary<TestEngine, element_type>(
            creator(element_type, ashape.dynamic_shape), input, expected, ashape, tol);
    }

    template <typename TestEngine,
              element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(Creator creator,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    ArgShape ashape = {},
                    ArgTolerance tol = {})
    {
        std::vector<value_type> expected;
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<TestEngine, element_type, Creator>(creator, input, expected, ashape, tol);
    }
}
