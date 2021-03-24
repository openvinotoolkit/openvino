// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, logical_or)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto B = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::v1::LogicalOr>(A, B), ParameterVector{A, B});

    std::vector<char> a{1, 0, 1, 1, 1, 0, 1, 0};
    std::vector<char> b{0, 0, 1, 0, 0, 1, 1, 0};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<char>({a, b});
    test_case.add_expected_output<char>(shape, {1, 0, 1, 1, 1, 1, 1, 0});
    test_case.run();
}
