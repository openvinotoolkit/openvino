// Co pyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, prelu)
{
    Shape shape{3, 2};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{0, 0.5, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{0, 3, -1, 1, -1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_shared_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{0.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{-1, 3, -1, 1, -0.5, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{-0.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{1, 3, 1, 1, 0.5, 0});
    test_case.run();
}
