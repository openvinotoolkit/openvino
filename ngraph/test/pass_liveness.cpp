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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "pass/liveness.hpp"

#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
namespace ng = ngraph;

TEST(liveness, constant)
{
    Shape shape{1};
    auto c = op::Constant::create(element::i32, shape, {5});
    auto f = make_shared<Function>(make_shared<op::Negative>(c), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(f);

    auto tmp = f->get_ordered_ops();
    vector<shared_ptr<Node>> sorted{tmp.begin(), tmp.end()};
    ASSERT_EQ(3, sorted.size());
    EXPECT_EQ(0, sorted[0]->liveness_new_list.size());
    EXPECT_EQ(0, sorted[0]->liveness_free_list.size());

    // op::Negative is live on output to op::Result
    // op::Negative is new
    EXPECT_EQ(1, sorted[1]->liveness_new_list.size());
    EXPECT_EQ(0, sorted[1]->liveness_free_list.size());

    // op::Negative is live on input to op::Result
    EXPECT_EQ(0, sorted[2]->liveness_new_list.size());
    // op::Negative is freed
    EXPECT_EQ(1, sorted[2]->liveness_free_list.size());
}
