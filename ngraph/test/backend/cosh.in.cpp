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

NGRAPH_TEST(${BACKEND_NAME}, cosh_float) {
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), ParameterVector{A});

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    vector<float> expected;
    for (float f : input) {
        expected.push_back(coshf(f));
    }

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(input);
    test_case.add_expected_output<float>(shape, expected);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, cosh_int) {
    Shape shape{5};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({1, 5, 2, 3, 3});
    test_case.add_expected_output<int32_t>(shape, {2, 74, 4, 10, 10});
    test_case.run();
}
