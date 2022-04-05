// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(pass_manager, add) {
    pass::Manager pass_manager;

    auto graph = make_test_graph();
    size_t node_count = 0;
    traverse_nodes(graph, [&](shared_ptr<Node> /* node */) {
        node_count++;
    });
    pass_manager.run_passes(graph);
    auto sorted = graph->get_ordered_ops();
    EXPECT_EQ(node_count, sorted.size());
    EXPECT_TRUE(validate_list(sorted));
}

namespace {
class DummyPass : public pass::FunctionPass {
public:
    DummyPass() {}
    bool run_on_function(std::shared_ptr<ngraph::Function> /* f */) override {
        return false;
    }
};
}  // namespace
