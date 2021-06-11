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

namespace
{
    template <typename T>
    std::shared_ptr<ngraph::Function> getTestFunction(ngraph::element::Type_t inputs_type,
                                                      Shape data_shape,
                                                      Shape axis_shape,
                                                      Shape split_lenghts_shape,
                                                      std::vector<T> axis_value,
                                                      std::vector<T> split_lenghts_value)
    {
        const auto data = make_shared<op::Parameter>(inputs_type, data_shape);
        const auto axis = op::Constant::create(inputs_type, axis_shape, axis_value);
        const auto split_lengths =
            op::Constant::create(inputs_type, split_lenghts_shape, split_lenghts_value);
        const auto variadic_split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);
        return make_shared<Function>(variadic_split, ParameterVector{data});
    }
} // namespace

NGRAPH_TEST(${BACKEND_NAME}, variadic_split_1d)
{
    const Shape data_shape{10};
    const std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const Shape axis_shape{1};
    const std::vector<int32_t> axis_value{0};
    const Shape split_lenghts_shape{3};
    const std::vector<int32_t> split_lenghts{5, 3, 2};

    auto test_case = test::TestCase<TestEngine>(getTestFunction(
        element::i32, data_shape, axis_shape, split_lenghts_shape, axis_value, split_lenghts));

    test_case.add_input(data_shape, data);
    test_case.add_expected_output<int32_t>(Shape{5}, {1, 2, 3, 4, 5});
    test_case.add_expected_output<int32_t>(Shape{3}, {6, 7, 8});
    test_case.add_expected_output<int32_t>(Shape{2}, {9, 10});
    test_case.run();
}
