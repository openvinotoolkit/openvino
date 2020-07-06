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
        class GraphRewriteBase;
        class GraphRewrite;
        class RecurrentGraphRewrite;
        class MatcherPass;
    }

    using graph_rewrite_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
    using recurrent_graph_rewrite_callback =
        std::function<bool(ngraph::pattern::RecurrentMatcher& m)>;
    using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
}

class NGRAPH_API ngraph::pass::MatcherPass : public ngraph::pass::PassBase
{
public:
    MatcherPass() = default;

    explicit MatcherPass(const std::string& name,
                         const handler_callback& handler,
                         const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE)
        : PassBase()
        , m_handler(handler)
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

    std::vector<std::shared_ptr<ngraph::Node>> get_new_nodes() const { return m_new_nodes; }
protected:
    void register_matcher(const std::shared_ptr<pattern::Matcher>& m,
                          const ngraph::graph_rewrite_callback& callback,
                          const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE);

private:
    handler_callback m_handler;
    std::vector<std::shared_ptr<ngraph::Node>> m_new_nodes;
};

class NGRAPH_API ngraph::pass::GraphRewriteBase : public ngraph::pass::FunctionPass
{
public:
    /// \brief Add an arbitrary handler for nodes
    /// \param name The name of the handler
    /// \param handler Function responsible for deciding if the graph should be changed and making
    /// the changes. Returns true if changes are made.
    void add_handler(const std::string& name,
                     std::function<bool(const std::shared_ptr<Node>& node)> handler,
                     const PassPropertyMask& property);

protected:
    GraphRewriteBase()
        : FunctionPass()
    {
        // Being explicit:
        // Setting REQUIRE_STATIC_SHAPE to false because we will check if each
        // callback needs static shape during run_on_function().
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, false);
    }

    bool is_enabled(const std::string& name) const;

    std::vector<std::shared_ptr<ngraph::pass::MatcherPass>> m_matchers;
};

/// \brief GraphRewrite (in tandem with \sa Matcher) performs transformations on specified patterns
///
/// Graph rewrite pass essentially allows pass users to rewrite parts of the
/// input graph in any way they want. Fusion is one example of graph rewrite that
/// fuses multiple ops together. At a high-level users of the pass need to
/// specify 2 things: 1) which ops to fuse (via \sa Matcher, and 2) how to create new op(s) from
/// the existing ops by providing a callback to \p Matcher object
/// Patterns can be added by using \sa add_matcher
/// Callbacks should use \sa replace_node to transform matched sub graphs

class NGRAPH_API ngraph::pass::GraphRewrite : public ngraph::pass::GraphRewriteBase
{
public:
    GraphRewrite() = default;

    explicit GraphRewrite(const std::shared_ptr<MatcherPass>& pass)
        : GraphRewriteBase()
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
                     const PassPropertyMask& property);

    // TODO: This interface may deprecate after all passes are refactored.
    void add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                     const ngraph::graph_rewrite_callback& callback);

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    bool m_enable_shape_inference = false;
};

class NGRAPH_API ngraph::pass::RecurrentGraphRewrite : public ngraph::pass::GraphRewriteBase
{
public:
    RecurrentGraphRewrite(size_t num_iters = 10)
        : GraphRewriteBase()
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
};
