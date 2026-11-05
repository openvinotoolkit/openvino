// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

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
    return std::make_shared<ov::Model>(ov::OutputVector{divide}, ov::ParameterVector{data});
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
    return std::make_shared<ov::Model>(ov::OutputVector{divide}, ov::ParameterVector{data});
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

namespace {

std::shared_ptr<MatcherPass> make_recording_pass(const std::shared_ptr<Node>& root, NodeVector& visited) {
    auto matcher = std::make_shared<pattern::Matcher>(root, "RecordingPass");
    return std::make_shared<MatcherPass>("RecordingPass", matcher, [&visited](const std::shared_ptr<Node>& node) {
        visited.push_back(node);
        return false;
    });
}

}  // namespace

TEST(GraphRewriteTest, OrRootFiltersUnrelatedTypes) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<op::v1::Divide>(visited.front()));
}

TEST(GraphRewriteTest, NestedOrRootVisitsSharedTypeOnce) {
    auto model = get_model();
    auto divide = pattern::wrap_type<op::v1::Divide>();
    auto root = (divide | pattern::wrap_type<op::v0::Relu>()) | divide;
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<op::v1::Divide>(visited.front()));
}

TEST(GraphRewriteTest, OrRootVisitsOverlappingBaseAndDerivedTypesOnce) {
    auto model = get_derived_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<PrivateDivide>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<PrivateDivide>(visited.front()));
}

TEST(GraphRewriteTest, OrRootSupportsOperationPatterns) {
    auto model = get_model();
    auto divide = std::make_shared<op::v1::Divide>(pattern::any_input(), pattern::any_input());
    auto root = divide | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<op::v1::Divide>(visited.front()));
}

TEST(GraphRewriteTest, OrRootWithUntypedAlternativeKeepsFallback) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::any_input();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    EXPECT_EQ(visited, model->get_ordered_ops());
}

TEST(GraphRewriteTest, OrRootWithOptionalAlternativeKeepsFallback) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v0::Relu>() | pattern::optional<op::v0::Tanh>(pattern::any_input());
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    EXPECT_EQ(visited, model->get_ordered_ops());
}

TEST(GraphRewriteTest, OrRootPreservesMatcherOrder) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v0::Relu>() | pattern::wrap_type<op::v1::Divide>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<op::v1::Divide>(visited.front()));
    EXPECT_EQ(count_ops_of_type<op::v0::Relu>(model), 1);
}

TEST(GraphRewriteTest, OrRootStopsAfterSuccessfulMatcher) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    EXPECT_TRUE(visited.empty());
    EXPECT_EQ(count_ops_of_type<op::v0::Relu>(model), 1);
}

TEST(GraphRewriteTest, DisabledOrRootDoesNotRun) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.get_pass_config()->disable<MatcherPass>();
    pass.run_on_model(model);

    EXPECT_TRUE(visited.empty());
}

TEST(BackwardGraphRewriteOrderTest, OrRootPreservesReverseOrder) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Parameter>();
    NodeVector visited;
    BackwardGraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 2);
    EXPECT_TRUE(ov::is_type<op::v1::Divide>(visited[0]));
    EXPECT_EQ(visited[1], model->get_parameters().front());
}

TEST(GraphRewriteTest, OrRootWithMultiOutputAlternative) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<op::v1::Split>(input, axis, 2);
    auto model = std::make_shared<Model>(split->outputs(), ParameterVector{input});
    auto split_pattern = std::make_shared<op::v1::Split>(pattern::any_input(), axis, 2);
    auto root = split_pattern | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    EXPECT_EQ(visited, NodeVector{split});
}

TEST(GraphRewriteTest, OrRootRecursesIntoSubgraphs) {
    auto then_body = get_model();
    auto else_body = get_model();
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 1, 2});
    auto condition = op::v0::Constant::create(element::boolean, Shape{}, {true});
    auto if_op = std::make_shared<op::v8::If>(condition);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(input, then_body->get_parameters().front(), else_body->get_parameters().front());
    auto output = if_op->set_output(then_body->get_results().front(), else_body->get_results().front());
    auto model = std::make_shared<Model>(OutputVector{output}, ParameterVector{input});
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Relu>();
    NodeVector visited;
    GraphRewrite pass;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    ASSERT_EQ(visited.size(), 2);
    EXPECT_EQ(visited[0], then_body->get_results().front()->input_value(0).get_node_shared_ptr());
    EXPECT_EQ(visited[1], else_body->get_results().front()->input_value(0).get_node_shared_ptr());
}

TEST(GraphRewriteTest, OrRootProcessesRegisteredNodes) {
    auto model = get_model();
    auto root = pattern::wrap_type<op::v1::Divide>() | pattern::wrap_type<op::v0::Relu>();
    auto matcher = std::make_shared<pattern::Matcher>(root, "DivideToRelu");
    GraphRewrite pass;
    std::shared_ptr<MatcherPass> rewrite;
    rewrite = pass.add_matcher(std::make_shared<MatcherPass>(matcher, [&rewrite](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!ov::is_type<op::v1::Divide>(node))
            return false;
        auto relu = rewrite->register_new_node<op::v0::Relu>(node->input_value(0));
        ov::replace_node(node, relu);
        return true;
    }));
    NodeVector visited;
    pass.add_matcher(make_recording_pass(root, visited));
    pass.run_on_model(model);

    EXPECT_EQ(visited, NodeVector{model->get_results().front()->input_value(0).get_node_shared_ptr()});
    ASSERT_EQ(visited.size(), 1);
    EXPECT_TRUE(ov::is_type<op::v0::Relu>(visited.front()));
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
