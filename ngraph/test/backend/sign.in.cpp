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

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({1, -2, 0, -4.8f, 4.8f, -0.0f});
    test_case.add_expected_output<float>(shape,
                                         {1, -1, 0, -1, 1, 0});
    test_case.run();
}
