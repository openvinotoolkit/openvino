// Copyright (C) 2021 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, unsqueeze)
{
    auto data_node = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto squeeze = make_shared<op::v0::Unsqueeze>(data_node, axes_node);

    auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 1, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, unsqueeze_negative_axis)
{
    auto data_node = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, -1});
    auto squeeze = make_shared<op::v0::Unsqueeze>(data_node, axes_node);

    auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = test::TestCase<TestEngine>(function);

    auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 2, 1}, data);
    test_case.run();
}
