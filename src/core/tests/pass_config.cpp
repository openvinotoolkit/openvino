// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_tools.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ::testing;
using namespace std;
using namespace ov;
using namespace ov::pass;

class RenameReLU : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RenameReLU");
    RenameReLU() : MatcherPass() {
        auto relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>();
        ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
            auto relu = m.get_match_root();
            relu->set_friendly_name("renamed");
            return false;
        };

        auto m = std::make_shared<pass::pattern::Matcher>(relu, "RenameReLU");
        this->register_matcher(m, callback);
    }
};

class RenameSigmoid : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RenameSigmoid");
    RenameSigmoid() : MatcherPass() {
        auto sigmoid = pattern::wrap_type<ov::op::v0::Sigmoid>();
        ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
            auto sigmoid = m.get_match_root();
            sigmoid->set_friendly_name("renamed");
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(sigmoid, "RenameSigmoid");
        this->register_matcher(m, callback);
    }
};

class TestModelPass : public pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TestModelPass");

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        pass::Manager manager(get_pass_config());

        manager.register_pass<RenameReLU, false /*disabled by default*/>();
        manager.register_pass<RenameSigmoid>();

        manager.run_passes(f);
        return true;
    }
};

class TestGraphRewritePass : public pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TestGraphRewritePass");
    TestGraphRewritePass() {
        add_matcher<RenameReLU, false /*disabled by default*/>();
        add_matcher<RenameSigmoid>();
    }
};

static std::tuple<std::shared_ptr<Model>, std::shared_ptr<Node>, std::shared_ptr<Node>> get_test_function() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
    auto relu = std::make_shared<ov::op::v0::Relu>(data);
    relu->set_friendly_name("relu");
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(relu);
    sigmoid->set_friendly_name("sigmoid");
    auto f = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{data});
    return std::tuple<std::shared_ptr<Model>, std::shared_ptr<Node>, std::shared_ptr<Node>>(f, relu, sigmoid);
}

TEST(PassConfig, EnableDisablePasses1) {
    std::shared_ptr<Model> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestModelPass>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses2) {
    std::shared_ptr<Model> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestModelPass>();

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

TEST(PassConfig, EnableDisablePasses3) {
    std::shared_ptr<Model> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestModelPass>();

    auto pass_config = manager.get_pass_config();
    pass_config->enable<RenameReLU>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "renamed");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses4) {
    std::shared_ptr<Model> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestModelPass>();

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

TEST(PassConfig, EnableDisablePasses5) {
    std::shared_ptr<Model> f;
    std::shared_ptr<Node> relu, sigmoid;
    std::tie(f, relu, sigmoid) = get_test_function();

    pass::Manager manager;
    manager.register_pass<TestGraphRewritePass>();
    manager.run_passes(f);

    ASSERT_EQ(relu->get_friendly_name(), "relu");
    ASSERT_EQ(sigmoid->get_friendly_name(), "renamed");
}

TEST(PassConfig, EnableDisablePasses6) {
    std::shared_ptr<Model> f;
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

TEST(PassConfig, EnableDisablePasses7) {
    std::shared_ptr<Model> f;
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

TEST(PassConfig, EnableDisablePasses8) {
    std::shared_ptr<Model> f;
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

TEST(PassConfig, EnableDisablePasses9) {
    std::shared_ptr<Model> f;
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

class TestNestedMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TestNestedMatcher");
    TestNestedMatcher() : MatcherPass() {
        auto any_op = pattern::any_input();
        ov::matcher_pass_callback callback = [this](pattern::Matcher& m) {
            auto root = m.get_match_root();
            auto pass_config = this->get_pass_config();
            if (ov::as_type_ptr<op::v0::Relu>(root) && !pass_config->is_disabled<RenameReLU>()) {
                auto pass = std::make_shared<RenameReLU>();
                pass->set_pass_config(pass_config);
                pass->apply(root);
            } else if (ov::as_type_ptr<op::v0::Sigmoid>(root) && !pass_config->is_disabled<RenameSigmoid>()) {
                auto pass = std::make_shared<RenameSigmoid>();
                pass->set_pass_config(pass_config);
                pass->apply(root);
            }
            return false;
        };

        auto m = std::make_shared<pass::pattern::Matcher>(any_op, "TestNestedMatcher");
        this->register_matcher(m, callback);
    }
};

class TestNestedGraphRewrite : public pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TestNestedGraphRewrite");
    TestNestedGraphRewrite() {
        add_matcher<TestNestedMatcher>();
    }
};

TEST(PassConfig, EnableDisablePasses10) {
    std::shared_ptr<Model> f;
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

TEST(PassConfig, EnableDisablePasses11) {
    std::shared_ptr<Model> f;
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
