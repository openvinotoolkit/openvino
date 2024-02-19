// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
using matcher_pass_callback = std::function<bool(pass::pattern::Matcher& m)>;
using graph_rewrite_callback = std::function<bool(pass::pattern::Matcher& m)>;
using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
namespace pass {
/// \brief Register openvino node pointers into container.
/// Can create and/or add existing node pointers into register
class NodeRegistry {
public:
    /// \brief Make new node and add it to register.
    ///
    /// \tparam T     Node type.
    /// \tparam Args  Node ctor args types.
    ///
    /// \param args   New node ctor arguments.
    /// \return Shared pointer to node of type T.
    template <typename T, class... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        return add(node);
    }

    /// \brief Add node to register
    ///
    /// \tparam T  Node type.
    ///
    /// \param node  Node to add
    ///
    /// \return Shared pointer to new node added of type T.
    template <typename T>
    std::shared_ptr<T> add(const std::shared_ptr<T>& node) {
        m_nodes.push_back(node);
        return node;
    }

    /// \brief Get nodes container.
    ///
    /// \return Const reference to nodes container.
    const std::vector<std::shared_ptr<Node>>& get() const {
        return m_nodes;
    }

    /// \brief Clear register.
    void clear() {
        m_nodes.clear();
    }

private:
    std::vector<std::shared_ptr<Node>> m_nodes;  //!< Stores added nodes.
};

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
/// or has ov::pass::pattern::op::WrapType. That will help GraphRewrite to execute matcher
/// passes more
/// efficient.
/// \ingroup ov_pass_cpp_api
class OPENVINO_API MatcherPass : public PassBase {
public:
    OPENVINO_RTTI("ov::pass::MatcherPass");

    MatcherPass() = default;

    MatcherPass(const MatcherPass&) = delete;
    MatcherPass& operator=(const MatcherPass&) = delete;

    explicit MatcherPass(const std::string& name,
                         const std::shared_ptr<pattern::Matcher>& m,
                         const handler_callback& handler,
                         const PassPropertyMask& property = PassProperty::CHANGE_DYNAMIC_STATE)
        : PassBase(),
          m_handler(handler),
          m_matcher(m) {
        set_name(name);
        set_property(property, true);
    }

    MatcherPass(const std::shared_ptr<pattern::Matcher>& m, const matcher_pass_callback& callback) : PassBase() {
        register_matcher(m, callback);
    }

    bool apply(std::shared_ptr<ov::Node> node);

    template <typename T, class... Args>
    std::shared_ptr<T> register_new_node(Args&&... args) {
        return m_new_nodes.make<T>(std::forward<Args>(args)...);
    }

    template <typename T>
    std::shared_ptr<T> register_new_node(const std::shared_ptr<T>& node) {
        return m_new_nodes.add(node);
    }

    std::shared_ptr<ov::Node> register_new_node_(const std::shared_ptr<ov::Node>& node) {
        return register_new_node(node);
    }

    const std::vector<std::shared_ptr<ov::Node>>& get_new_nodes() {
        return m_new_nodes.get();
    }

    void clear_new_nodes() {
        m_new_nodes.clear();
    }

    std::shared_ptr<pattern::Matcher> get_matcher() {
        return m_matcher;
    }

protected:
    void register_matcher(const std::shared_ptr<pattern::Matcher>& m,
                          const matcher_pass_callback& callback,
                          const PassPropertyMask& property);

    void register_matcher(const std::shared_ptr<pattern::Matcher>& m, const matcher_pass_callback& callback);

private:
    handler_callback m_handler;
    std::shared_ptr<pattern::Matcher> m_matcher;
    NodeRegistry m_new_nodes;
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
/// or has ov::pattern::op::WrapType. That will help GraphRewrite to execute matcher
/// passes more
/// efficient.
/// \ingroup ov_pass_cpp_api
class OPENVINO_API GraphRewrite : public ModelPass {
public:
    OPENVINO_RTTI("ov::pass::GraphRewrite");

    GraphRewrite() = default;

    explicit GraphRewrite(const std::shared_ptr<MatcherPass>& pass) : ModelPass() {
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
              typename std::enable_if<std::is_base_of<pass::MatcherPass, T>::value, bool>::type = true>
    std::shared_ptr<T> add_matcher(Args&&... args) {
        static_assert(std::is_base_of<pass::MatcherPass, T>::value, "pass not derived from MatcherPass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_config = get_pass_config();
        pass->set_pass_config(pass_config);
        if (!Enabled && !pass_config->is_enabled<T>()) {
            pass_config->disable<T>();
        }
        m_matchers.push_back(pass);
        return pass;
    }

    /// \brief Register passes from GraphRewrite class that contains sequence of matcher
    /// passes registered in its ctor.
    /// For example:
    ///
    ///    class ov::pass::LinFusions: public ov::pass::GraphRewrite {
    ///    public:
    ///         OPENVINO_RTTI("LinFusion");
    ///         Fusions() {
    ///             add_matcher<ov::pass::AddFusion>();
    ///             add_matcher<ov::pass::MulFusion>();
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
              typename std::enable_if<std::is_base_of<pass::GraphRewrite, T>::value, bool>::type = true>
    void add_matcher(Args&&... args) {
        static_assert(std::is_base_of<pass::GraphRewrite, T>::value, "pass not derived from GraphRewrite");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        auto pass_config = get_pass_config();

        for (auto& matcher : pass->m_matchers) {
            pass->set_pass_config(pass_config);
            m_matchers.push_back(matcher);
        }
    }

    std::shared_ptr<MatcherPass> add_matcher(const std::shared_ptr<MatcherPass>& pass) {
        auto pass_config = get_pass_config();
        pass->set_pass_config(pass_config);
        m_matchers.push_back(pass);
        return pass;
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    void set_pass_config(const std::shared_ptr<PassConfig>& pass_config) override;

protected:
    bool apply_matcher_passes(std::shared_ptr<Model> f, std::deque<std::weak_ptr<Node>> nodes_to_run);

    bool m_enable_shape_inference = false;

    std::vector<std::shared_ptr<ov::pass::MatcherPass>> m_matchers;
};

class OPENVINO_API BackwardGraphRewrite : public GraphRewrite {
public:
    OPENVINO_RTTI("ov::pass::BackwardGraphRewrite");

    BackwardGraphRewrite() = default;

    explicit BackwardGraphRewrite(const std::shared_ptr<MatcherPass>& pass) : GraphRewrite(pass) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
}  // namespace pass
}  // namespace ov
