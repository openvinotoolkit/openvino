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

// Tolerance can be specified in two forms (number of bits in mantissa or float-point
// threshold), corresponding to two different comparing methods implemented in TestCase
// classes. Following class acts as an encapsulation on these two options without introducing
// too many overloadings on the function requiring such argument.
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

struct ArgTolerance
{
    BitTolerance bit = {};
    FloatTolerance fp = {};

    ArgTolerance() = default;
    ArgTolerance(const float tolerance)
        : fp(tolerance)
    {
    }
    ArgTolerance(const int tolerance)
        : bit(tolerance)
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
    ngraph::PartialShape dynamic_shape;
    ngraph::Shape static_shape;
    bool initialzed;
    ArgShape(const ngraph::PartialShape& dynamic_shape, const ngraph::Shape& static_shape)
        : dynamic_shape(dynamic_shape)
        , static_shape(static_shape)
        , initialzed(true)
    {
    }
    ArgShape(const ngraph::Shape& static_shape)
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
          ngraph::element::Type_t element_type,
          typename value_type = ngraph::fundamental_type_for<element_type>>
void test_unary(std::shared_ptr<ngraph::Function> f,
                const std::vector<value_type>& test_input,
                const std::vector<value_type>& test_expected,
                ArgShape ashape = {},
                ArgTolerance tol = {})
{
    if (!ashape.initialzed)
    {
        ashape = ArgShape(ngraph::Shape({test_input.size()}));
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

    auto test_case = ngraph::test::TestCase<TestEngine>(f);
    test_case.template add_input<value_type>(ashape.static_shape, input);
    test_case.template add_expected_output<value_type>(ashape.static_shape, expected);

    if (tol.bit.in_use)
        test_case.run(tol.bit.value);
    else if (tol.fp.in_use)
        test_case.run_with_tolerance_as_fp(tol.fp.value);
    else
        test_case.run();
}

// overloading support passing reference function instead of expected results
template <typename TestEngine,
          ngraph::element::Type_t element_type,
          typename value_type = ngraph::fundamental_type_for<element_type>>
void test_unary(std::shared_ptr<ngraph::Function> f,
                const std::vector<value_type>& input,
                value_type (*get_expect)(value_type),
                ArgShape ashape = {},
                ArgTolerance tol = {})
{
    std::vector<value_type> expected;
    for (value_type x : input)
        expected.push_back(get_expect(x));

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
using UnaryFuncCreator = std::function<std::shared_ptr<ngraph::Function>(
    const ngraph::element::Type&, const ngraph::PartialShape&)>;

template <
    typename OpType,
    typename... Args,
    typename std::enable_if<std::is_base_of<ngraph::op::Op, OpType>::value, bool>::type = true>
UnaryFuncCreator unary_func(Args&&... args)
{
    return [args...](const ngraph::element::Type& ele_type, const ngraph::PartialShape& pshape) {
        auto X = std::make_shared<ngraph::op::Parameter>(ele_type, pshape);
        auto Y = std::make_shared<OpType>(X, args...);
        return std::make_shared<ngraph::Function>(Y, ngraph::ParameterVector{X});
    };
}

template <typename TestEngine,
          ngraph::element::Type_t element_type,
          typename Creator,
          typename value_type = ngraph::fundamental_type_for<element_type>>
void test_unary(Creator creator,
                const std::vector<value_type>& input,
                const std::vector<value_type>& expected,
                ArgShape ashape = {},
                ArgTolerance tol = {})
{
    if (!ashape.initialzed)
    {
        ashape = ArgShape(ngraph::Shape({input.size()}));
    }
    test_unary<TestEngine, element_type>(
        creator(element_type, ashape.dynamic_shape), input, expected, ashape, tol);
}

template <typename TestEngine,
          ngraph::element::Type_t element_type,
          typename Creator,
          typename value_type = ngraph::fundamental_type_for<element_type>>
void test_unary(Creator creator,
                const std::vector<value_type>& input,
                value_type (*get_expect)(value_type),
                ArgShape ashape = {},
                ArgTolerance tol = {})
{
    std::vector<value_type> expected;
    for (value_type x : input)
        expected.push_back(get_expect(x));

    test_unary<TestEngine, element_type, Creator>(creator, input, expected, ashape, tol);
}
