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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_ceil)
{
    Shape in_shape{1, 1, 4, 4};
    Shape out_shape{1, 1, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto f = make_shared<Function>(make_shared<op::v1::MaxPool>(A, {2, 2}, {0, 0}
                                                                {0, 0}, {3, 3}, 
                                                                ngraph::op::RoundingType::CEIL,
                                                                ngraph::op::RoundingType::CEIL), ParameterVector{A});

    std::vector<float> a{{{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    }}}

    std::vector<float> result{{{
        {11, 12},
        {15, 15}
    }}}

    auto test_case = test::TestCase<TestEngine>(f);
    test_case_add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}