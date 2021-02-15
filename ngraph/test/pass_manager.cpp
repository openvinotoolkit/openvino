//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(pass_manager, add)
{
    pass::Manager pass_manager;

    auto graph = make_test_graph();
    size_t node_count = 0;
    traverse_nodes(graph, [&](shared_ptr<Node> /* node */) { node_count++; });
    pass_manager.run_passes(graph);
    auto sorted = graph->get_ordered_ops();
    EXPECT_EQ(node_count, sorted.size());
    EXPECT_TRUE(validate_list(sorted));
}

namespace
{
    class DummyPass : public pass::FunctionPass
    {
    public:
        DummyPass()
            : FunctionPass()
        {
        }
        bool run_on_function(std::shared_ptr<ngraph::Function> /* f */) override { return false; }
    };
}
