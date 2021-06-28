// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util/unary_test.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, sqrt_basic)
{
    test::make_unary_test<TestEngine, op::Sqrt, element::f32>(Shape{2, 3})
        .test({16, 4, 81, 100, 10000, 0}, {4, 2, 9, 10, 100, 0});
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_negative_inputs)
{
    test::make_unary_test<TestEngine, op::Sqrt, element::f32>(Shape{4}).test({-1, 4, -81, 100},
                                                                             {NAN, 2, NAN, 10});
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_integral_inputs)
{
    test::make_unary_test<TestEngine, op::Sqrt, element::i32>(Shape{2, 7})
        .test({4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889},
              {2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333});
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_floating_inputs)
{
    test::make_unary_test<TestEngine, op::Sqrt, element::f32>(Shape{2, 7})
        .test({4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88},
              {2.,
               2.6457512,
               3.,
               3.1622777,
               8.944272,
               7.4161983,
               2.5,
               0.94868326,
               4.830114,
               15.264338,
               16.,
               21.766697,
               32.,
               333.33});
}
