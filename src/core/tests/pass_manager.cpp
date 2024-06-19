// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

using namespace ov;
using namespace std;

namespace {

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

class TestMatcherPassTrue : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TestMatcherPassTrue");
    TestMatcherPassTrue() : MatcherPass() {
        auto any_input = ov::pass::pattern::any_input();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_input, "TestMatcherPassTrue");
        this->register_matcher(m, callback);
    }
};

class TestMatcherPassFalse : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TestMatcherPassFalse");
    TestMatcherPassFalse() : MatcherPass() {
        auto any_input = ov::pass::pattern::any_input();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_input, "TestMatcherPassFalse");
        this->register_matcher(m, callback);
    }
};

class TestModelPassTrue : public pass::ModelPass {
public:
    OPENVINO_RTTI("TestModelPassTrue");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        return true;
    }
};

class TestModelPassFalse : public pass::ModelPass {
public:
    OPENVINO_RTTI("TestModelPassFalse");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        return false;
    }
};

class TestValidate : public pass::Validate {
public:
    OPENVINO_RTTI("TestValidate");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        m_applied = true;
        return pass::Validate::run_on_model(f);
    }

    bool is_applied() const {
        return m_applied;
    }

private:
    bool m_applied = false;
};

class TestValidate2 : public TestValidate {};

class TestManager : public pass::Manager {
public:
    bool is_validation_applied() {
        bool applied = false;
        bool is_init = true;
        for (const auto& pass : m_pass_list) {
            auto validate_2 = std::dynamic_pointer_cast<TestValidate2>(pass);
            auto validate = std::dynamic_pointer_cast<TestValidate>(pass);
            if (validate && !validate_2) {
                if (is_init) {
                    is_init = false;
                    applied = validate->is_applied();
                }
                applied = applied && validate->is_applied();
            }
        }
        return applied;
    }

    bool is_2nd_validation_applied() {
        for (const auto& pass : m_pass_list) {
            if (auto validate = std::dynamic_pointer_cast<TestValidate2>(pass)) {
                return validate->is_applied();
            }
        }
    }
};
}  // namespace

TEST(pass_manager, add) {
    pass::Manager pass_manager;

    auto graph = make_test_graph();
    size_t node_count = 0;
    ov::traverse_nodes(graph, [&](shared_ptr<Node> /* node */) {
        node_count++;
    });
    pass_manager.run_passes(graph);
    auto sorted = graph->get_ordered_ops();
    EXPECT_EQ(node_count, sorted.size());
    EXPECT_TRUE(validate_list(sorted));
}

TEST(pass_manager, passes_not_applied) {
    TestManager pass_manager;
    pass_manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate>();
    const auto res = pass_manager.run_passes(graph);

    EXPECT_FALSE(res);
    EXPECT_FALSE(pass_manager.is_validation_applied());
}

TEST(pass_manager, model_pass_applied) {
    TestManager pass_manager;
    pass_manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassTrue>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate>();
    const auto res = pass_manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(pass_manager.is_validation_applied());
}

TEST(pass_manager, matcher_pass_applied) {
    TestManager pass_manager;
    pass_manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    pass_manager.register_pass<TestMatcherPassTrue>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate>();
    const auto res = pass_manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(pass_manager.is_validation_applied());
}

TEST(pass_manager, two_validations) {
    TestManager pass_manager;
    pass_manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    pass_manager.register_pass<TestMatcherPassTrue>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate2>();
    pass_manager.register_pass<TestMatcherPassTrue>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestMatcherPassFalse>();
    pass_manager.register_pass<TestModelPassFalse>();
    pass_manager.register_pass<TestValidate>();
    const auto res = pass_manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(pass_manager.is_validation_applied());
    EXPECT_FALSE(pass_manager.is_2nd_validation_applied());
}