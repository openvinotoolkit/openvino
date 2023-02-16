// Copyright (C) 2021-2023 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, squeeze)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_default_axes)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_no_axes)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto squeeze = make_shared<op::Squeeze>(data_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}
