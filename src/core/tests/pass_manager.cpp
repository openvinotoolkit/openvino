// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"

using namespace ngraph;
using namespace std;

OPENVINO_SUPPRESS_DEPRECATED_START

std::shared_ptr<ov::Model> make_test_graph() {
    auto arg_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_5 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});

    auto t0 = std::make_shared<ov::op::v1::Add>(arg_0, arg_1);
    auto t1 = std::make_shared<ov::op::v0::MatMul>(t0, arg_2);
    auto t2 = std::make_shared<ov::op::v1::Multiply>(t0, arg_3);

    auto t3 = std::make_shared<ov::op::v1::Add>(t1, arg_4);
    auto t4 = std::make_shared<ov::op::v1::Add>(t2, arg_5);

    auto r0 = std::make_shared<ov::op::v1::Add>(t3, t4);

    auto m = std::make_shared<ov::Model>(r0, ov::ParameterVector{arg_0, arg_1, arg_2, arg_3, arg_4, arg_5});

    return m;
}

// This function traverses the vector of ops and verifies that each op's dependencies (its inputs)
// is located earlier in the vector. That is enough to be valid
bool validate_list(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    bool rc = true;
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
        auto node_tmp = *it;
        ov::NodeVector dependencies_tmp;
        for (auto& val : node_tmp->input_values())
            dependencies_tmp.emplace_back(val.get_node_shared_ptr());
        std::vector<ov::Node*> dependencies;

        for (std::shared_ptr<ov::Node> n : dependencies_tmp) {
            dependencies.push_back(n.get());
        }
        auto tmp = it;
        for (tmp++; tmp != nodes.rend(); tmp++) {
            auto dep_tmp = *tmp;
            auto found = find(dependencies.begin(), dependencies.end(), dep_tmp.get());
            if (found != dependencies.end()) {
                dependencies.erase(found);
            }
        }
        if (dependencies.size() > 0) {
            rc = false;
        }
    }
    return rc;
}

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
    bool run_on_model(const std::shared_ptr<ngraph::Function>& /* f */) override {
        return false;
    }
};
}  // namespace
