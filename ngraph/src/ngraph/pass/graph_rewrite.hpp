//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
        class RecurrentGraphRewrite;
        class MatcherPass;
    }

    using matcher_pass_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
    using graph_rewrite_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
    using recurrent_graph_rewrite_callback =
        std::function<bool(ngraph::pattern::RecurrentMatcher& m)>;
    using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
}

/// \brief MatcherPass is a basic block for pattern based transformations. It describes pattern and
/// action that is applied if pattern is matched.
///
/// MatcherPass consists of Matcher and matcher_pass_callback that needs to be implemented and
/// finally registered by using \sa register_matcher. MatcherPass can be executed on node within
/// \sa apply method. To run matcher pass on Function use GraphRewrite.
/// In addition MatcherPass provides a way for adding new operations into GraphRewrite execution
/// queue. That means that operations that were created inside transformation callback can be added
/// for matching. To register node use \sa register_new_node method. GraphRewrite automatically
/// takes registered nodes and put them to execution queue. If multiple nodes were register make
/// sure that they were registered in topological order.
/// Note: when implementing pattern for Matcher make sure that root node is an operation from opset
/// or has ngraph::pattern::op::WrapType. That will help GraphRewrite to execute matcher passes more
/// efficient.

class NGRAPH_API ngraph::pass::MatcherPass : public ngraph::pass::PassBase
{
public:
    MatcherPass() = default;

    MatcherPass(const MatcherPass&) = delete;
    MatcherPass& operator=(const MatcherPass&) = delete;

    explicit MatcherPass(const std::string& name,
                         const std::shared_ptr<pattern::Matcher>& m,
                         const handler_callback& handler,
                         const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE)
        : PassBase()
        , m_handler(handler)
        , m_matcher(m)
    {
        set_name(name);
        set_property(property, true);
    }

    bool apply(std::shared_ptr<ngraph::Node> node);

    template <typename T, class... Args>
    std::shared_ptr<T> register_new_node(Args&&... args)
    {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        m_new_nodes.push_back(node);
        return node;
    }

    std::vector<std::shared_ptr<ngraph::Node>> get_new_nodes() { return m_new_nodes; }
    void clear_new_nodes() { m_new_nodes.clear(); }
    std::shared_ptr<pattern::Matcher> get_matcher() { return m_matcher; }
protected:
    void register_matcher(const std::shared_ptr<pattern::Matcher>& m,
                          const ngraph::graph_rewrite_callback& callback,
                          const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE);

private:
    handler_callback m_handler;
    std::shared_ptr<pattern::Matcher> m_matcher;
    std::vector<std::shared_ptr<ngraph::Node>> m_new_nodes;
};

/// \brief GraphRewrite is a container for MatcherPasses that allows to run them on Function in
/// efficient way
///
/// Graph rewrite pass is used for matcher passes execution on Function.
/// To register MatcherPass use \sa add_matcher<T>(args) method where T is a MatcherPass class.
/// As a default algorithm graph rewrite pass traverse Function in topological order and applies
/// registered matcher passes for each node. But if all registered matcher passes have type based
/// root node in Matcher pattern then efficient mechanism is used to execute them.
/// Matcher pattern root is type based if it's operation from opset or pattern::op::WrapType.
/// Note: when implementing pattern for Matcher make sure that root node is an operation from opset
/// or has ngraph::pattern::op::WrapType. That will help GraphRewrite to execute matcher passes more
/// efficient.

class NGRAPH_API ngraph::pass::GraphRewrite : public ngraph::pass::FunctionPass
{
public:
    GraphRewrite() = default;

    explicit GraphRewrite(const std::shared_ptr<MatcherPass>& pass)
        : FunctionPass()
    {
        m_matchers.push_back(pass);
    }

    template <typename T, class... Args>
    std::shared_ptr<T> add_matcher(Args&&... args)
    {
        static_assert(std::is_base_of<pass::MatcherPass, T>::value,
                      "pass not derived from MatcherPass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        m_matchers.push_back(pass);
        return pass;
    }

    void add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                     const ngraph::graph_rewrite_callback& callback,
                     const PassPropertyMask& property) NGRAPH_DEPRECATED("Use MatcherPass instead");

    void add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                     const ngraph::graph_rewrite_callback& callback)
        NGRAPH_DEPRECATED("Use MatcherPass instead");

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    bool m_enable_shape_inference = false;

    std::vector<std::shared_ptr<ngraph::pass::MatcherPass>> m_matchers;
};

class NGRAPH_API ngraph::pass::RecurrentGraphRewrite : public ngraph::pass::FunctionPass
{
public:
    RecurrentGraphRewrite(size_t num_iters = 10)
        : FunctionPass()
        , m_num_iters(num_iters)
    {
    }

    void add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                     const ngraph::recurrent_graph_rewrite_callback& callback,
                     const PassPropertyMask& property);

    // TODO: This interface may deprecate after all passes are refactored.
    void add_matcher(const std::shared_ptr<pattern::RecurrentMatcher>& m,
                     const ngraph::recurrent_graph_rewrite_callback& callback);

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);

private:
    size_t m_num_iters;

    std::vector<std::shared_ptr<ngraph::pass::MatcherPass>> m_matchers;
};
