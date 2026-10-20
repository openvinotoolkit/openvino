// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ::testing;
using namespace std;
using namespace ov;
using namespace ov::pass;

inline std::shared_ptr<Model> get_model();

namespace {
class RootFilterPass : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RootFilterPass");
    RootFilterPass(const std::shared_ptr<pattern::Matcher>& matcher,
                   const matcher_pass_callback& callback,
                   bool filter = true,
                   bool static_only = false) {
        register_matcher(matcher, callback);
        if (!filter)
            set_property(PassProperty::MATCHER_ROOT_TYPE_FILTER, false);
        if (static_only)
            set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
};

class PermissiveMatcher : public pattern::Matcher {
public:
    using pattern::Matcher::Matcher;
    bool match_value(const Output<Node>& pattern_value, const Output<Node>& graph_value) override {
        get_pattern_value_map()[pattern_value.get_node_shared_ptr()] = graph_value;
        return true;
    }
};

class PermissiveRelu : public op::v0::Relu {
public:
    using op::v0::Relu::Relu;
    bool match_value(pattern::Matcher*, const Output<Node>&, const Output<Node>&) override {
        return true;
    }
};

class PermissiveWrapType : public pattern::op::WrapType {
public:
    PermissiveWrapType() : WrapType(std::vector<NodeTypeInfo>{op::v0::Relu::get_type_info_static()}) {}
    bool match_value(pattern::Matcher*, const Output<Node>&, const Output<Node>&) override {
        return true;
    }
};

class DerivedRelu : public op::v0::Relu {
public:
    OPENVINO_RTTI("RootFilterDerivedRelu", "test", op::v0::Relu);
    using op::v0::Relu::Relu;
};

void add_unknown_root(const std::shared_ptr<GraphRewrite>& rewrite) {
    rewrite->add_matcher(
        std::make_shared<MatcherPass>(std::make_shared<pattern::Matcher>(pattern::any_input()), [](pattern::Matcher&) {
            return false;
        }));
}

std::shared_ptr<pattern::Matcher> make_or_matcher() {
    return std::make_shared<pattern::Matcher>(std::make_shared<pattern::op::Or>(
        OutputVector{pattern::wrap_type<op::v0::Relu>(), pattern::wrap_type<op::v0::Tanh>()}));
}
}  // namespace

TEST(GraphRewriteRootFilter, RejectClearsMatcherStateAndNewNodes) {
    const auto model = get_model();
    const auto matcher = make_or_matcher();
    auto rewrite = std::make_shared<GraphRewrite>();
    auto pass = std::make_shared<RootFilterPass>(matcher, [](pattern::Matcher&) {
        ADD_FAILURE() << "No Relu or Tanh exists in the model";
        return false;
    });
    rewrite->add_matcher(pass);
    add_unknown_root(rewrite);
    const auto stale = std::make_shared<op::v0::Relu>(model->get_parameters().front());
    ASSERT_TRUE(matcher->match(stale->output(0)));
    pass->register_new_node(stale);
    ASSERT_FALSE(rewrite->run_on_model(model));
    EXPECT_TRUE(matcher->get_pattern_value_map().empty());
    EXPECT_TRUE(matcher->get_pattern_value_maps().empty());
    EXPECT_TRUE(matcher->get_matched_values().empty());
    EXPECT_TRUE(matcher->get_symbols().empty());
    EXPECT_TRUE(pass->get_new_nodes().empty());
}

TEST(GraphRewriteRootFilter, CustomHandlerAndMatcherKeepLegacyExecution) {
    const auto model = get_model();
    auto rewrite = std::make_shared<GraphRewrite>();
    size_t handlers = 0;
    size_t callbacks = 0;
    auto root = pattern::wrap_type<op::v0::Relu>();
    rewrite->add_matcher(std::make_shared<MatcherPass>("custom",
                                                       std::make_shared<pattern::Matcher>(root),
                                                       [&](const std::shared_ptr<Node>&) {
                                                           ++handlers;
                                                           return false;
                                                       }));
    rewrite->add_matcher(
        std::make_shared<RootFilterPass>(std::make_shared<PermissiveMatcher>(root), [&](pattern::Matcher&) {
            ++callbacks;
            return false;
        }));
    add_unknown_root(rewrite);
    ASSERT_FALSE(rewrite->run_on_model(model));
    EXPECT_EQ(handlers, model->get_ordered_ops().size());
    EXPECT_EQ(callbacks, model->get_ordered_ops().size());
}

TEST(GraphRewriteRootFilter, CustomOperationAndPatternKeepLegacyExecution) {
    const auto model = get_model();
    auto rewrite = std::make_shared<GraphRewrite>();
    for (const auto& root : NodeVector{std::make_shared<PermissiveRelu>(pattern::any_input()),
                                       std::make_shared<PermissiveWrapType>(),
                                       std::make_shared<pattern::op::Or>(
                                           OutputVector{pattern::wrap_type<op::v0::Relu>(), pattern::any_input()})}) {
        size_t callbacks = 0;
        rewrite = std::make_shared<GraphRewrite>();
        rewrite->add_matcher(
            std::make_shared<RootFilterPass>(std::make_shared<pattern::Matcher>(root), [&](pattern::Matcher&) {
                ++callbacks;
                return false;
            }));
        add_unknown_root(rewrite);
        ASSERT_FALSE(rewrite->run_on_model(model));
        EXPECT_EQ(callbacks, model->get_ordered_ops().size());
    }
}

TEST(GraphRewriteRootFilter, OrBranchAndRegistrationOrderMatchUnfilteredExecution) {
    for (const bool backward : {false, true}) {
        std::vector<std::string> reference;
        for (const bool filter : {false, true}) {
            const auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
            const auto relu = std::make_shared<DerivedRelu>(data);
            const auto tanh = std::make_shared<op::v0::Tanh>(relu);
            const auto model = std::make_shared<Model>(OutputVector{tanh}, ParameterVector{data});
            std::shared_ptr<GraphRewrite> rewrite =
                backward ? std::make_shared<BackwardGraphRewrite>() : std::make_shared<GraphRewrite>();
            const auto first = pattern::wrap_type<op::v0::Relu, op::v0::Tanh>();
            const auto second = pattern::wrap_type<op::v0::Relu, op::v0::Tanh>();
            const auto matcher =
                std::make_shared<pattern::Matcher>(std::make_shared<pattern::op::Or>(OutputVector{first, second}));
            std::vector<std::string> order;
            rewrite->add_matcher(std::make_shared<RootFilterPass>(
                matcher,
                [&](pattern::Matcher& m) {
                    EXPECT_EQ(m.get_pattern_value_map().count(first), 1);
                    EXPECT_EQ(m.get_pattern_value_map().count(second), 0);
                    order.push_back("first:" + std::string(m.get_match_root()->get_type_name()));
                    return false;
                },
                filter));
            rewrite->add_matcher(std::make_shared<RootFilterPass>(
                make_or_matcher(),
                [&](pattern::Matcher& m) {
                    order.push_back("second:" + std::string(m.get_match_root()->get_type_name()));
                    return false;
                },
                filter));
            add_unknown_root(rewrite);
            ASSERT_FALSE(rewrite->run_on_model(model));
            ASSERT_EQ(order.size(), 4);
            if (!filter)
                reference = order;
            else
                EXPECT_EQ(order, reference);
        }
    }
}

TEST(GraphRewriteRootFilter, PatternMutationFallsBackConservatively) {
    const auto model = get_model();
    auto rewrite = std::make_shared<GraphRewrite>();
    const auto root = std::make_shared<pattern::op::Or>(OutputVector{pattern::wrap_type<op::v0::Relu>()});
    const auto matcher = std::make_shared<pattern::Matcher>(root);
    size_t callbacks = 0;
    rewrite->add_matcher(std::make_shared<RootFilterPass>(matcher, [&](pattern::Matcher&) {
        ++callbacks;
        return false;
    }));
    root->input(0).replace_source_output(pattern::any_input()->output(0));
    ASSERT_FALSE(rewrite->run_on_model(model));
    EXPECT_EQ(callbacks, model->get_ordered_ops().size());
}

TEST(GraphRewriteRootFilter, AnyOutputPreservesMultiOutputMatching) {
    for (const bool filter : {false, true}) {
        const auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        const auto k = op::v0::Constant::create(element::i32, Shape{}, {2});
        const auto topk = std::make_shared<op::v1::TopK>(data, k, 0, "max", "value");
        const auto model =
            std::make_shared<Model>(OutputVector{topk->output(0), topk->output(1)}, ParameterVector{data});
        const auto pattern_topk = std::make_shared<op::v1::TopK>(pattern::any_input(), k, 0, "max", "value");
        const auto matcher = std::make_shared<pattern::Matcher>(pattern_topk);
        auto rewrite = std::make_shared<GraphRewrite>();
        size_t callbacks = 0;
        rewrite->add_matcher(std::make_shared<RootFilterPass>(
            matcher,
            [&](pattern::Matcher& m) {
                ++callbacks;
                EXPECT_EQ(m.get_match_root(), topk);
                return false;
            },
            filter));
        add_unknown_root(rewrite);
        ASSERT_FALSE(rewrite->run_on_model(model));
        EXPECT_EQ(callbacks, 1);
    }
}

TEST(GraphRewriteRootFilter, CommutativeOperandOrderMatchesUnfilteredExecution) {
    for (const bool filter : {false, true}) {
        const auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
        const auto constant = op::v0::Constant::create(element::f32, Shape{2}, {1, 2});
        const auto add = std::make_shared<op::v1::Add>(constant, data);
        const auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{data});
        const auto constant_pattern = pattern::wrap_type<op::v0::Constant>();
        const auto data_pattern = pattern::wrap_type<op::v0::Parameter>();
        const auto add_pattern = pattern::wrap_type<op::v1::Add>({data_pattern, constant_pattern});
        auto rewrite = std::make_shared<GraphRewrite>();
        size_t callbacks = 0;
        rewrite->add_matcher(std::make_shared<RootFilterPass>(
            std::make_shared<pattern::Matcher>(add_pattern),
            [&](pattern::Matcher& m) {
                ++callbacks;
                EXPECT_EQ(m.get_pattern_value_map().at(data_pattern).get_node_shared_ptr(), data);
                EXPECT_EQ(m.get_pattern_value_map().at(constant_pattern).get_node_shared_ptr(), constant);
                return false;
            },
            filter));
        add_unknown_root(rewrite);
        ASSERT_FALSE(rewrite->run_on_model(model));
        EXPECT_EQ(callbacks, 1);
    }
}

TEST(GraphRewriteRootFilter, NewNodesAreProcessedBeforeRemainingModelNodes) {
    for (const bool filter : {false, true}) {
        const auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
        const auto relu = std::make_shared<op::v0::Relu>(data);
        const auto model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data});
        auto rewrite = std::make_shared<GraphRewrite>();
        std::vector<std::string> order;
        auto pass = std::make_shared<RootFilterPass>(
            make_or_matcher(),
            [&](pattern::Matcher& m) {
                order.emplace_back(m.get_match_root()->get_type_name());
                return false;
            },
            filter);
        std::shared_ptr<RootFilterPass> replace;
        replace = std::make_shared<RootFilterPass>(
            std::make_shared<pattern::Matcher>(pattern::wrap_type<op::v0::Relu>()),
            [&](pattern::Matcher& m) {
                auto tanh = replace->register_new_node<op::v0::Tanh>(m.get_match_root()->input_value(0));
                replace_node(m.get_match_root(), tanh);
                return true;
            },
            filter);
        rewrite->add_matcher(replace);
        rewrite->add_matcher(pass);
        add_unknown_root(rewrite);
        ASSERT_TRUE(rewrite->run_on_model(model));
        EXPECT_EQ(order, std::vector<std::string>{"Tanh"});
        EXPECT_TRUE(replace->get_new_nodes().empty());
        EXPECT_EQ(count_ops_of_type<op::v0::Relu>(model), 0);
        EXPECT_EQ(count_ops_of_type<op::v0::Tanh>(model), 1);
    }
}

TEST(GraphRewriteRootFilter, DisabledAndStaticOnlyPassesKeepTheirGuards) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    const auto relu = std::make_shared<op::v0::Relu>(data);
    const auto model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data});
    auto rewrite = std::make_shared<GraphRewrite>();
    rewrite->add_matcher(std::make_shared<RootFilterPass>(
        make_or_matcher(),
        [](pattern::Matcher&) {
            ADD_FAILURE() << "Static-only pass must not run on a dynamic model";
            return false;
        },
        true,
        true));
    add_unknown_root(rewrite);
    ASSERT_FALSE(rewrite->run_on_model(model));
    rewrite->get_pass_config()->disable<RootFilterPass>();
    ASSERT_FALSE(rewrite->run_on_model(model));
}

TEST(GraphRewriteRootFilter, DeepOrFallsBackWithoutUnboundedFilterRecursion) {
    const auto model = get_model();
    std::shared_ptr<Node> root = pattern::any_input();
    for (size_t i = 0; i < 20; ++i)
        root = std::make_shared<pattern::op::Or>(OutputVector{root});
    auto rewrite = std::make_shared<GraphRewrite>();
    size_t callbacks = 0;
    rewrite->add_matcher(
        std::make_shared<RootFilterPass>(std::make_shared<pattern::Matcher>(root), [&](pattern::Matcher&) {
            ++callbacks;
            return false;
        }));
    ASSERT_FALSE(rewrite->run_on_model(model));
    EXPECT_EQ(callbacks, model->get_ordered_ops().size());
}

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
