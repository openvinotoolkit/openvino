// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph
{
    using matcher_pass_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
    using graph_rewrite_callback = std::function<bool(ngraph::pattern::Matcher& m)>;
    using recurrent_graph_rewrite_callback =
        std::function<bool(ngraph::pattern::RecurrentMatcher& m)>;
    using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
    namespace pass
    {
        /// \brief MatcherPass is a basic block for pattern based transformations. It describes
        /// pattern and
        /// action that is applied if pattern is matched.
        ///
        /// MatcherPass consists of Matcher and matcher_pass_callback that needs to be implemented
        /// and
        /// finally registered by using \sa register_matcher. MatcherPass can be executed on node
        /// within
        /// \sa apply method. To run matcher pass on Function use GraphRewrite.
        /// In addition MatcherPass provides a way for adding new operations into GraphRewrite
        /// execution
        /// queue. That means that operations that were created inside transformation callback can
        /// be added
        /// for matching. To register node use \sa register_new_node method. GraphRewrite
        /// automatically
        /// takes registered nodes and put them to execution queue. If multiple nodes were register
        /// make
        /// sure that they were registered in topological order.
        /// Note: when implementing pattern for Matcher make sure that root node is an operation
        /// from opset
        /// or has ngraph::pattern::op::WrapType. That will help GraphRewrite to execute matcher
        /// passes more
        /// efficient.

        class NGRAPH_API MatcherPass : public ngraph::pass::PassBase
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            MatcherPass() = default;

            MatcherPass(const MatcherPass&) = delete;
            MatcherPass& operator=(const MatcherPass&) = delete;

            explicit MatcherPass(
                const std::string& name,
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

            const std::vector<std::shared_ptr<ngraph::Node>>& get_new_nodes()
            {
                return m_new_nodes;
            }
            void clear_new_nodes() { m_new_nodes.clear(); }
            std::shared_ptr<pattern::Matcher> get_matcher() { return m_matcher; }

        protected:
            void register_matcher(
                const std::shared_ptr<pattern::Matcher>& m,
                const ngraph::graph_rewrite_callback& callback,
                const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE);

        private:
            handler_callback m_handler;
            std::shared_ptr<pattern::Matcher> m_matcher;
            std::vector<std::shared_ptr<ngraph::Node>> m_new_nodes;
        };

        /// \brief GraphRewrite is a container for MatcherPasses that allows to run them on Function
        /// in
        /// efficient way
        ///
        /// Graph rewrite pass is used for matcher passes execution on Function.
        /// To register MatcherPass use \sa add_matcher<T>(args) method where T is a MatcherPass
        /// class.
        /// As a default algorithm graph rewrite pass traverse Function in topological order and
        /// applies
        /// registered matcher passes for each node. But if all registered matcher passes have type
        /// based
        /// root node in Matcher pattern then efficient mechanism is used to execute them.
        /// Matcher pattern root is type based if it's operation from opset or
        /// pattern::op::WrapType.
        /// Note: when implementing pattern for Matcher make sure that root node is an operation
        /// from opset
        /// or has ngraph::pattern::op::WrapType. That will help GraphRewrite to execute matcher
        /// passes more
        /// efficient.

        class NGRAPH_API GraphRewrite : public ngraph::pass::FunctionPass
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            GraphRewrite() = default;

            explicit GraphRewrite(const std::shared_ptr<MatcherPass>& pass)
                : FunctionPass()
            {
                m_matchers.push_back(pass);
            }

            /// \brief Register given transformation class type to GraphRewrite execution list
            /// All registered transformations will be executed in a single graph traversal.
            /// Example below show the basic usage of pass::GraphRewrite
            ///
            ///     pass::Manager manager;
            ///     auto anchor = manager.register_pass<GraphRewrite>();
            ///     anchor->add_matcher<MatcherPassA>();
            ///     anchor->add_matcher<MatcherPassB>();
            ///     anchor->set_name("CommonMatchers");
            ///     manager.run_passes(f);
            ///
            /// For some purposes transformation can be registered and disabled by default.
            ///
            ///     anchor->add_matcher<MatcherPassB, false>();
            ///
            /// \return shared_ptr to the transformation instance
            template <typename T,
                      bool Enabled = true,
                      class... Args,
                      typename std::enable_if<std::is_base_of<pass::MatcherPass, T>::value,
                                              bool>::type = true>
            std::shared_ptr<T> add_matcher(Args&&... args)
            {
                static_assert(std::is_base_of<pass::MatcherPass, T>::value,
                              "pass not derived from MatcherPass");
                auto pass = std::make_shared<T>(std::forward<Args>(args)...);
                auto pass_config = get_pass_config();
                pass->set_pass_config(pass_config);
                if (!Enabled && !pass_config->is_enabled<T>())
                {
                    pass_config->disable<T>();
                }
                m_matchers.push_back(pass);
                return pass;
            }

            /// \brief Register passes from GraphRewrite class that contains sequence of matcher
            /// passes registered in its ctor.
            /// For example:
            ///
            ///    class ngraph::pass::LinFusions: public ngraph::pass::GraphRewrite {
            ///    public:
            ///         NGRAPH_RTTI_DECLARATION;
            ///         Fusions() {
            ///             add_matcher<ngraph::pass::AddFusion>();
            ///             add_matcher<ngraph::pass::MulFusion>();
            ///         }
            ///     };
            ///
            ///     pass::Manager manager;
            ///     auto anchor = manager.register_pass<GraphRewrite>();
            ///     anchor->add_matcher<LinFusions>();
            ///     anchor->add_matcher<OtherFusions>();
            ///     anchor->set_name("CommonFusions");
            ///     manager.run_passes(f);
            ///
            /// In this case all matcher passes from LinFusions pass will be united with other
            /// registered matchers.
            template <typename T,
                      class... Args,
                      typename std::enable_if<std::is_base_of<pass::GraphRewrite, T>::value,
                                              bool>::type = true>
            void add_matcher(Args&&... args)
            {
                static_assert(std::is_base_of<pass::GraphRewrite, T>::value,
                              "pass not derived from GraphRewrite");
                auto pass = std::make_shared<T>(std::forward<Args>(args)...);
                auto pass_config = get_pass_config();

                for (auto& matcher : pass->m_matchers)
                {
                    pass->set_pass_config(pass_config);
                    m_matchers.push_back(matcher);
                }
            }

            NGRAPH_DEPRECATED("Use MatcherPass instead")
            void add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                             const ngraph::graph_rewrite_callback& callback,
                             const PassPropertyMask& property);

            NGRAPH_DEPRECATED("Use MatcherPass instead")
            void add_matcher(const std::shared_ptr<pattern::Matcher>& m,
                             const ngraph::graph_rewrite_callback& callback);

            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

            void set_pass_config(const std::shared_ptr<PassConfig>& pass_config) override;

        protected:
            bool apply_matcher_passes(std::shared_ptr<Function> f,
                                      std::deque<std::shared_ptr<Node>> nodes_to_run);

            bool m_enable_shape_inference = false;

            std::vector<std::shared_ptr<ngraph::pass::MatcherPass>> m_matchers;
        };

        class NGRAPH_API BackwardGraphRewrite : public ngraph::pass::GraphRewrite
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            BackwardGraphRewrite() = default;

            explicit BackwardGraphRewrite(const std::shared_ptr<MatcherPass>& pass)
                : GraphRewrite(pass)
            {
            }

            bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
        };

        class NGRAPH_API RecurrentGraphRewrite : public ngraph::pass::FunctionPass
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
    } // namespace pass
} // namespace ngraph
