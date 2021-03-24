// Copyright (C) 2018-2021 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, aliased_output)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::v1::Add>(A, B);
    auto D = make_shared<op::v1::Multiply>(A, B);
    auto E = op::Constant::create(element::f32, shape, {1, 2, 3, 4});
    auto f = make_shared<Function>(NodeVector{C, C, D, D, C, E, E}, ParameterVector{A, B});

    vector<float> a{0, 1, 2, 3};
    vector<float> b{1, 2, 3, 4};
    vector<float> expectedC{1, 3, 5, 7};
    vector<float> expectedD{0, 2, 6, 12};
    vector<float> expectedE{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedD);
    test_case.add_expected_output<float>(shape, expectedD);
    test_case.add_expected_output<float>(shape, expectedC);
    test_case.add_expected_output<float>(shape, expectedE);
    test_case.add_expected_output<float>(shape, expectedE);
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}
