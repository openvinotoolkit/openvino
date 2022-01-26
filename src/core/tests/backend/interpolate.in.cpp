// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/execute_tools.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
static string s_device = test::backend_name_to_device("${BACKEND_NAME}");

NGRAPH_TEST(${BACKEND_NAME}, interpolate_down_scales_const_linear) {
    Shape input_shape{1, 1, 2, 4};
    Shape output_shape{1, 1, 1, 2};
    op::v0::InterpolateAttrs attrs;
    attrs.axes = AxisSet{0, 1, 2, 3};
    attrs.mode = "linear";
    attrs.align_corners = false;
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto output_shape_input = op::v0::Constant::create(element::i64, {4}, {1, 1, 1, 2});
    std::vector<float> intput_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    vector<float> expected_output{1.0f, 2.66666651f};

    auto interpolate = make_shared<op::v0::Interpolate>(input, output_shape_input, attrs);
    auto f = make_shared<Function>(interpolate, ParameterVector{input});

    auto test_case = test::TestCase(f, s_device);
    test_case.add_input(intput_data);
    test_case.add_expected_output(expected_output);
    test_case.run();
}
