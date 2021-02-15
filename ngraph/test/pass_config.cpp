// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <util/test_tools.hpp>

using namespace ::testing;
using namespace std;
using namespace ngraph;

class RenameReLU : public ngraph::pass::MatcherPass
{
public:
    NGRAPH_RTTI_DECLARATION;
    RenameReLU()
        : MatcherPass()
    {
        auto relu = pattern::wrap_type<opset3::Relu>();
        ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
            auto relu = m.get_match_root();
            relu->set_friendly_name("renamed");
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(relu, "RenameReLU");
        this->register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(RenameReLU, "RenameReLU", 0);

class RenameSigmoid : public ngraph::pass::MatcherPass
{
public:
    NGRAPH_RTTI_DECLARATION;
    RenameSigmoid()
        : MatcherPass()
    {
        auto sigmoid = pattern::wrap_type<opset3::Sigmoid>();
        ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
            auto sigmoid = m.get_match_root();
            sigmoid->set_friendly_name("renamed");
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(sigmoid, "RenameSigmoid");
        this->register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(RenameSigmoid, "RenameSigmoid", 0);

class TestFunctionPass : public ngraph::pass::FunctionPass
{
public:
    NGRAPH_RTTI_DECLARATION;

    bool run_on_function(std::shared_ptr<Function> f) override
    {
        pass::Manager manager(get_pass_config());

        manager.register_pass<RenameReLU, false /*disabled by default*/>();
        manager.register_pass<RenameSigmoid>();

        manager.run_passes(f);
        return true;
    }
};

NGRAPH_RTTI_DEFINITION(TestFunctionPass, "TestFunctionPass", 0);

class TestGraphRewritePass : public ngraph::pass::GraphRewrite
{
public:
    NGRAPH_RTTI_DECLARATION;
    TestGraphRewritePass()
    {
        add_matcher<RenameReLU, false /*disabled by default*/>();
        add_matcher<RenameSigmoid>();
    }
};

NGRAPH_RTTI_DEFINITION(TestGraphRewritePass, "TestGraphRewritePass", 0);

std::tuple<std::shared_ptr<Function>, std::shared_ptr<Node>, std::shared_ptr<Node>>
    get_test_function()
{
    auto data =
        std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
    auto relu = std::make_shared<ngraph::opset3::Relu>(data);
    relu->set_friendly_name("relu");
    auto sigmoid = std::make_shared<ngraph::opset3::Sigmoid>(relu);
    sigmoid->set_friendly_name("sigmoid");
    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid},
                                                ngraph::ParameterVector{data});
    return std::tuple<std::shared_ptr<Function>, std::shared_ptr<Node>, std::shared_ptr<Node>>(
        f, relu, sigmoid);
}

TEST(PassConfig, EnableDisablePasses1)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestFunctionPass>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses2)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestFunctionPass>();

    auto pass_config = manager.get_pass_config();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameSigmoid>();
    pass_config->enable<RenameReLU>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses3)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestFunctionPass>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses4)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestFunctionPass>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameSigmoid>();
    manager.run_passes(f);
    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses5)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestGraphRewritePass>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses6)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestGraphRewritePass>();

    auto pass_config = manager.get_pass_config();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameSigmoid>();
    pass_config->enable<RenameReLU>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses7)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestGraphRewritePass>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses8)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestGraphRewritePass>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses9)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<pass::GraphRewrite>();
    anchor->add_matcher<RenameReLU, false>();
    anchor->add_matcher<RenameSigmoid>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

class TestNestedMatcher : public ngraph::pass::MatcherPass
{
public:
    NGRAPH_RTTI_DECLARATION;
    TestNestedMatcher()
        : MatcherPass()
    {
        auto any_op = pattern::any_input();
        ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
            auto root = m.get_match_root();
            auto pass_config = this->get_pass_config();
            if (std::dynamic_pointer_cast<opset3::Relu>(root) &&
                !pass_config->is_disabled<RenameReLU>())
            {
                auto pass = std::make_shared<RenameReLU>();
                pass->set_pass_config(pass_config);
                pass->apply(root);
            }
            else if (std::dynamic_pointer_cast<opset3::Sigmoid>(root) &&
                     !pass_config->is_disabled<RenameSigmoid>())
            {
                auto pass = std::make_shared<RenameSigmoid>();
                pass->set_pass_config(pass_config);
                pass->apply(root);
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(any_op, "TestNestedMatcher");
        this->register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(TestNestedMatcher, "TestNestedMatcher", 0);

class TestNestedGraphRewrite : public pass::GraphRewrite
{
public:
    NGRAPH_RTTI_DECLARATION;
    TestNestedGraphRewrite() { add_matcher<TestNestedMatcher>(); }
};

NGRAPH_RTTI_DEFINITION(TestNestedGraphRewrite, "TestNestedGraphRewrite", 0);

TEST(PassConfig, EnableDisablePasses10)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestNestedGraphRewrite>();

    auto pass_config = manager.get_pass_config();
    pass_config->disable<RenameReLU>();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameReLU>();
    pass_config->enable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses11)
{
    std::shared_ptr<Function> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<pass::GraphRewrite>();
    anchor->add_matcher<TestNestedMatcher>();

    auto pass_config = manager.get_pass_config();
    pass_config->disable<RenameReLU>();
    pass_config->disable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "sigmoid");

    pass_config->enable<RenameReLU>();
    pass_config->enable<RenameSigmoid>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}
