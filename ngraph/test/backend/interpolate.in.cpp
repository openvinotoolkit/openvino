//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, interpolate_down_scales_const_linear)
{
    Shape input_shape{1, 1, 2, 4};
    Shape output_shape{1, 1, 1, 2};
    op::v0::InterpolateAttrs attrs;
    attrs.axes = AxisSet{0, 1, 2, 3};
    attrs.mode = "linear";
    attrs.align_corners = false;
    const auto input = make_shared<op::Parameter>(element::Type_t::f32, input_shape);
    const auto output_shape_input =
        op::v0::Constant::create(element::Type_t::i64, {4}, {1, 1, 1, 2});
    std::vector<float> intput_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    auto interpolate = make_shared<op::v0::Interpolate>(input, output_shape_input, attrs);
    auto f = make_shared<Function>(interpolate, ParameterVector{input});

    auto backend = runtime::Backend::create("IE_CPU");
    auto input_tensor = backend->create_tensor(element::Type_t::f32, input_shape);
    auto result_tensor = backend->create_tensor(element::Type_t::f32, output_shape);
    auto handle = backend->compile(f);
    copy_data(input_tensor, intput_data);

    handle->call_with_validate({result_tensor}, {input_tensor});

    vector<float> expected_output{1.0f, 2.66666651f};
    EXPECT_TRUE(test::all_close_f(expected_output, read_vector<float>(result_tensor)));
}
