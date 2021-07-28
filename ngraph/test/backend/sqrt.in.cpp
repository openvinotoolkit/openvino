// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/engine/test_engines.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, sqrt_basic)
{
    Shape shape{2, 3};
    auto input_param = make_shared<op::Parameter>(element::f32, shape);
    auto function =
        make_shared<Function>(make_shared<op::Sqrt>(input_param), ParameterVector{input_param});

    std::vector<float> input_data{16, 4, 81, 100, 10000, 0};
    std::vector<float> expected_result{4, 2, 9, 10, 100, 0};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_negative_inputs)
{
    Shape shape{4};
    auto input_param = make_shared<op::Parameter>(element::f32, shape);
    auto function =
        make_shared<Function>(make_shared<op::Sqrt>(input_param), ParameterVector{input_param});

    std::vector<float> input_data{-1, 4, -81, 100};
    std::vector<float> expected_result{NAN, 2, NAN, 10};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_integral_inputs)
{
    Shape shape{2, 7};
    auto input_param = make_shared<op::Parameter>(element::i32, shape);
    auto function =
        make_shared<Function>(make_shared<op::Sqrt>(input_param), ParameterVector{input_param});

    std::vector<int> input_data{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889};
    std::vector<int> expected_result{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<int>(input_data);
    test_case.add_expected_output<int>(shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt_floating_inputs)
{
    Shape shape{2, 7};
    auto input_param = make_shared<op::Parameter>(element::f32, shape);
    auto function =
        make_shared<Function>(make_shared<op::Sqrt>(input_param), ParameterVector{input_param});

    std::vector<float> input_data{
        4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88};
    std::vector<float> expected_result{2.,
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
                                       333.33};

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(shape, expected_result);
    test_case.run();
}
