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
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, reshape_v1)
{
    auto arg = std::make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);

    auto f = std::make_shared<Function>(NodeVector{reshape_v1}, ParameterVector{arg, pattern});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto arg_data = vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto pattern_data = vector<int64_t>{2, 2, 3};

    auto arg_tensor = backend->create_tensor(element::Type_t::i64, Shape{arg_data.size()});
    auto pattern_tensor = backend->create_tensor(element::Type_t::i64, Shape{pattern_data.size()});
    copy_data(arg_tensor, arg_data);
    copy_data(pattern_tensor, pattern_data);

    auto output = backend->create_dynamic_tensor(element::Type_t::i64, PartialShape::dynamic());
    ex->call_with_validate({output}, {arg_tensor, pattern_tensor});

    ASSERT_EQ(output->get_element_type(), element::Type_t::i64);
    EXPECT_EQ(read_vector<int64_t>(output),
              vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}
