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
#include "util/all_close.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

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

    template <element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    std::shared_ptr<Function> f,
                    const std::vector<value_type>& test_input,
                    const std::vector<value_type>& test_expected,
                    PartialShape dynamic_shape,
                    Shape static_shape,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        bool must_support_dynamic = dynamic_shape.is_dynamic();

        auto backend = runtime::Backend::create(backend_name, must_support_dynamic);

        auto create_output_tensor = [&]() {
            if (must_support_dynamic)
                return backend->create_dynamic_tensor(element_type, dynamic_shape);
            return backend->create_tensor(element_type, dynamic_shape.get_shape());
        };

        auto a = backend->create_tensor(element_type, static_shape);
        auto result = create_output_tensor();

        auto static_size = shape_size(static_shape);

        // expand or shrink the size of input & expected values
        std::vector<value_type> input;
        std::vector<value_type> expected;
        for (size_t i = 0; i < static_size; i++)
        {
            input.push_back(test_input[i % test_input.size()]);
            expected.push_back(test_expected[i % test_expected.size()]);
        }

        copy_data(a, input);

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a});

        auto actual = read_vector<value_type>(result);

        // size equility test
        EXPECT_EQ(actual.size(), static_size);
        EXPECT_EQ(result->get_shape(), static_shape);

        if (element::Type(element_type).is_real())
        {
            for (size_t i = 0; i < actual.size(); ++i)
            {
                auto e = expected[i];
                auto a = actual[i];
                if (std::abs(e - a) > atol + rtol * std::abs(e) || std::isinf(e) != std::isinf(a) ||
                    std::isnan(e) != std::isnan(a))
                {
                    ASSERT_TRUE(false) << "result " << a << " is not close to expected " << e
                                       << " at input " << input[i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < actual.size(); ++i)
                ASSERT_EQ(actual[i], expected[i]) << "at input " << input[i];
        }
    }

    template <element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    Creator creator,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    PartialShape pshape,
                    Shape sshape,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        test_unary<element_type>(backend_name,
                                 creator(element_type, pshape),
                                 input,
                                 expected,
                                 pshape,
                                 sshape,
                                 atol,
                                 rtol);
    }

    template <element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    Creator creator,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    PartialShape pshape,
                    Shape sshape,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        std::vector<value_type> expected;
        // in case input cannot be precisely expressed by value_type
        // converting them first to value_type then back to double
        // would generate result more close to expectation
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<element_type>(backend_name,
                                 creator(element_type, pshape),
                                 input,
                                 expected,
                                 pshape,
                                 sshape,
                                 atol,
                                 rtol);
    }

    template <element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    Creator creator,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        Shape sshape({input.size()});
        test_unary<element_type, Creator>(
            backend_name, creator, input, expected, PartialShape(sshape), sshape, atol, rtol);
    }

    template <element::Type_t element_type,
              typename Creator,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    Creator creator,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        Shape sshape({input.size()});
        test_unary<element_type, Creator>(
            backend_name, creator, input, func, PartialShape(sshape), sshape, atol, rtol);
    }

    template <element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    PartialShape pshape,
                    Shape sshape,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        std::vector<value_type> expected;
        // in case input cannot be precisely expressed by value_type
        // converting them first to value_type then back to double
        // would generate result more close to expectation
        for (value_type x : input)
            expected.push_back(func(x));

        test_unary<element_type>(backend_name, f, input, expected, pshape, sshape, atol, rtol);
    }

    template <element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    const std::vector<value_type>& expected,
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        Shape sshape({input.size()});
        test_unary<element_type>(
            backend_name, f, input, expected, PartialShape(sshape), sshape, atol, rtol);
    }

    template <element::Type_t element_type,
              typename value_type = fundamental_type_for<element_type>>
    void test_unary(std::string backend_name,
                    std::shared_ptr<Function> f,
                    const std::vector<value_type>& input,
                    value_type (*func)(value_type),
                    double atol = 1e-8,
                    double rtol = 1e-5)
    {
        Shape sshape({input.size()});
        test_unary<element_type>(
            backend_name, f, input, func, PartialShape(sshape), sshape, atol, rtol);
    }
}
