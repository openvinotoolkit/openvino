// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
