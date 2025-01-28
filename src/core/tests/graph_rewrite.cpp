// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/label.hpp"

using namespace ::testing;
using namespace std;
using namespace ov;
using namespace ov::pass;

class TestPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TestPass");
    TestPass() : MatcherPass() {
        auto divide = std::make_shared<ov::pass::pattern::op::Label>(element::f32,
                                                                     Shape{},
                                                                     pattern::has_class<ov::op::v1::Divide>());
        ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto relu = std::make_shared<ov::op::v0::Relu>(m.get_match_root()->input_value(0));
                ov::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

class GatherNodesPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatherNodesPass");
    GatherNodesPass(NodeVector& order) : MatcherPass() {
        ov::matcher_pass_callback callback = [&order](pattern::Matcher& m) {
            order.push_back(m.get_match_root());
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::any_input(), "GatherNodesPass");
        this->register_matcher(m, callback);
    }
};

class Anchor : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("Anchor");
    Anchor() : GraphRewrite() {}
};

inline std::shared_ptr<Model> get_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
    auto divide_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.5});
    auto divide = std::make_shared<ov::op::v1::Divide>(data, divide_constant);
    return std::make_shared<ov::Model>(ov::NodeVector{divide}, ov::ParameterVector{data});
}

inline ov::pass::param_callback get_callback() {
    return [](const std::shared_ptr<const Node>& node) -> bool {
        if (ov::as_type_ptr<const op::v1::Divide>(node)) {
            return true;
        } else {
            return false;
        }
    };
}

TEST(GraphRewriteOrderTest, MatcherPass) {
    auto f = get_model();

    NodeVector order;
    ov::pass::Manager m;
    auto pass = m.register_pass<pass::GraphRewrite>();
    pass->add_matcher<GatherNodesPass>(order);
    m.run_passes(f);

    ASSERT_EQ(order, f->get_ordered_ops());
}

TEST(BackwardGraphRewriteOrderTest, MatcherPass) {
    auto f = get_model();

    NodeVector order;
    ov::pass::Manager m;
    auto pass = m.register_pass<pass::BackwardGraphRewrite>();
    pass->add_matcher<GatherNodesPass>(order);
    m.run_passes(f);

    auto ref_order = f->get_ordered_ops();
    std::reverse(ref_order.begin(), ref_order.end());
    ASSERT_EQ(order, ref_order);
}

TEST(GraphRewriteTest, MatcherPassCallback) {
    auto f = get_model();

    Anchor anchor;
    anchor.add_matcher<TestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, GraphRewriteCallback) {
    auto f = get_model();

    Anchor anchor;
    anchor.add_matcher<TestPass>();
    anchor.set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallbackDeprecated) {
    auto f = get_model();

    pass::Manager manager;
    auto anchor = manager.register_pass<Anchor>();
    anchor->add_matcher<TestPass>();
    manager.get_pass_config()->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback) {
    auto f = get_model();

    pass::Manager manager;
    auto anchor = manager.register_pass<Anchor>();
    anchor->add_matcher<TestPass>();
    auto pass_config = manager.get_pass_config();
    pass_config->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback2) {
    auto f = get_model();

    pass::Manager manager;
    auto anchor = manager.register_pass<TestPass>();
    manager.get_pass_config()->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

class PrivateDivide : public ov::op::v1::Divide {
public:
    OPENVINO_OP("PrivateDivide", "test_opset", ov::op::v1::Divide);
    using ov::op::v1::Divide::Divide;
};

static std::shared_ptr<Model> get_derived_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
    auto divide_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.5});
    auto divide = std::make_shared<PrivateDivide>(data, divide_constant);
    return std::make_shared<ov::Model>(ov::NodeVector{divide}, ov::ParameterVector{data});
}

TEST(GraphRewriteTest, MatcherPassCallbackDerived) {
    auto f = get_derived_model();

    Anchor anchor;
    anchor.add_matcher<TestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

class TypeBasedTestPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TypeBasedTestPass");
    TypeBasedTestPass() : MatcherPass() {
        auto divide = std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::pass::pattern::op::Label>(),
                                                           std::make_shared<ov::pass::pattern::op::Label>());
        //        element::f32, Shape{}, pattern::has_class<op::v1::Divide>());
        ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto relu = std::make_shared<ov::op::v0::Relu>(m.get_match_root()->input_value(0));
                ov::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

class TypeBasedTestPassDerived : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TypeBasedTestPassDerived");
    TypeBasedTestPassDerived() : MatcherPass() {
        auto divide = std::make_shared<PrivateDivide>(std::make_shared<ov::pass::pattern::op::Label>(),
                                                      std::make_shared<ov::pass::pattern::op::Label>());
        ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto tanh = std::make_shared<ov::op::v0::Tanh>(m.get_match_root()->input_value(0));
                ov::replace_node(m.get_match_root(), tanh);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

TEST(GraphRewriteTest, TypeBasedMatcherPassCallback) {
    auto f = get_model();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassCallbackDerived) {
    auto f = get_derived_model();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassOrder1) {
    auto f = get_derived_model();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.add_matcher<TypeBasedTestPassDerived>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassOrder2) {
    auto f = get_derived_model();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPassDerived>()->set_callback(get_callback());
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Tanh>(f), 1);
}

TEST(PassConfigTest, Test1) {
    {
        auto f = get_model();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto f = get_model();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto f = get_model();

        auto pass_config = std::make_shared<ov::pass::PassConfig>();
        pass::Manager manager(pass_config);

        manager.register_pass<TestPass>();

        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto f = get_model();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = anchor->get_pass_config();
        pass_config->set_callback(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto f = get_model();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = anchor->get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto pass_config = std::make_shared<pass::PassConfig>();

        pass::Manager manager1(pass_config);
        pass::Manager manager2(pass_config);
        ASSERT_EQ(pass_config.use_count(), 3);
    }

    {
        auto f = get_model();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        pass_config->disable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 0);

        pass_config->enable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }

    {
        auto f = get_model();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        pass_config->disable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 0);

        pass_config->enable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<op::v0::Relu>(f), 1);
    }
}

class CheckConsumers : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CheckConsumers");
    CheckConsumers() {
        ov::matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
            auto node = m.get_match_root();
            auto consumers = [](Node* node) {
                int64_t cnt{0};
                for (auto output : node->outputs()) {
                    cnt += output.get_target_inputs().size();
                }
                if (ov::as_type<op::v0::Parameter>(node) || ov::as_type<op::v0::Result>(node)) {
                    cnt += 1;
                }
                return cnt;
            };
            /* The expected number of use_count() for Node is equal to the sum of next components:
             * 1. Each consumer holds a pointer to Output<Node> which holds a shared_ptr to Node
             * 2. pattern::Matcher object holds a shared_ptr to the matched node
             * 3. Local node variable increases use_counter
             * 4. Some GraphRewrite facilities
             */
            auto cnt = consumers(node.get());
            if (node.use_count() != cnt + 6) {
                OPENVINO_THROW("Wrong number of consumers");
            }

            NodeVector nodes;
            for (const auto& inputs : node->input_values()) {
                nodes.emplace_back(inputs.get_node_shared_ptr());
            }

            /* The expected number of use_count() for Node is equal to the sum of next components:
             * 1. Each consumer holds a pointer to Output<Node> which holds a shared_ptr to Node
             * 2. Local input_node variable increases use_counter
             */
            for (const auto& input_node : nodes) {
                if (input_node.use_count() != consumers(input_node.get()) + 1) {
                    OPENVINO_THROW("Wrong number of consumers");
                }
            }
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::any_input(), "CheckConsumers");
        this->register_matcher(m, callback);
    }
};

TEST(GraphRewriteTest, nodes_use_count) {
    auto f = get_model();
    pass::Manager m;
    m.register_pass<CheckConsumers>();
    OV_ASSERT_NO_THROW(m.run_passes(f));
}
