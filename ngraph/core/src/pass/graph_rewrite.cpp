// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <deque>
#include <iostream>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <regex>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "perf_counters.hpp"

using namespace std;
using namespace ngraph;

/* GraphRewrite algorithm:
 * GraphRewrite processes an input graph in an topological order(i.e. args before users)
 * Given the following graph:          Abs2
 *                                   /       \
 *                         Constant1         Add4 - Result5
 *                                   \      /
 *                                    Neg3
 *
 * The topological order would be : `Constant1`, `Abs2`, `Neg3`, `Add4`, `Result5`
 * Note, `Abs2` comes before `Neg3` as `Abs2`'s id = 2 is *less* than `Neg3`'s one (id = 3)
 * Next, GraphRewrite will invoke matchers passes registered in add_matcher order.
 * For example:
 *     ngraph::pass::GraphRewrite pass;
 *     pass.add_matcher<m1>();
 *     pass.add_matcher<m2>();
 *     pass.add_matcher<m3>();
 * Matcher passes will be called as follows: `m1`, `m2`, `m3`
 * Matchers should only replace nodes in the graph that come before the current root
 * node in the topological order. For example, if Matcher matches Neg3, it should only
 * replace nodes `Abs2` and `Constant1` if needed
 * This gives Matchers a nice cascading property. For example, if m1 folds `Abs2(Constant1)`
 * and `m2` folds `Neg3(Constant1)` when `m3` is called on `Add4` it will discover that
 * both `Abs2` and `Neg3` were already replaced by constants, so `Add4` will also be folded into
 * one.
 * If any matcher passes succeeds the rest of the matchers will **not** be called.
 * E.g. if `m1` succeeds and replaces `Abs2` with a new constant, nor `m2` or `m3` will be called
 * However, sometimes, you will need more than one fusion occur on the same node.
 * In this case, you need to register nodes in MatcherPass manually using register_new_node method.
 * GraphRewrite will automatically add this nodes in the beginning of execution queue.
 * If MatcherPass register more than one node make sure that this nodes are registered in
 * topological order. */

NGRAPH_RTTI_DEFINITION(ngraph::pass::GraphRewrite, "ngraph::pass::GraphRewrite", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::BackwardGraphRewrite, "ngraph::pass::BackwardGraphRewrite", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::MatcherPass, "ngraph::pass::MatcherPass", 0);

namespace ngraph
{
    namespace pass
    {
        namespace internal
        {
            PerfCounters& perf_counters_graph_rewrite()
            {
                static PerfCounters counters;
                return counters;
            }
        } // namespace internal
    }     // namespace pass
} // namespace ngraph

bool pass::BackwardGraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    // Initialize execution queue with nodes in topological order
    deque<std::shared_ptr<Node>> nodes_to_run;
    for (auto& node : f->get_ordered_ops())
    {
        nodes_to_run.emplace_front(node);
    }
    return apply_matcher_passes(f, std::move(nodes_to_run));
}

bool pass::GraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    // Initialize execution queue with nodes in topological order
    deque<std::shared_ptr<Node>> nodes_to_run;
    for (auto& node : f->get_ordered_ops())
    {
        nodes_to_run.emplace_back(node);
    }
    return apply_matcher_passes(f, std::move(nodes_to_run));
}

bool pass::GraphRewrite::apply_matcher_passes(shared_ptr<Function> f,
                                              deque<std::shared_ptr<Node>> nodes_to_run)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "pass::GraphRewrite::run_on_function");

    bool rewritten = false;
    const auto& pass_config = get_pass_config();

    // Check that all Matchers in MatcherPasses has type bases root node
    bool all_roots_has_type = true;
    std::unordered_map<NodeTypeInfo, std::vector<size_t>> type_to_matcher;
    for (size_t matcher_index = 0; matcher_index < m_matchers.size(); ++matcher_index)
    {
        // Skip passes that are disabled
        if (pass_config->is_disabled(m_matchers[matcher_index]->get_type_info()))
            continue;

        auto matcher = m_matchers[matcher_index]->get_matcher();
        if (!matcher)
        {
            all_roots_has_type = false;
            break;
        }

        auto root = matcher->get_pattern_value().get_node_shared_ptr();
        // pattern::op::AnyOutput operation automatically appends for multi output operations inside
        // Matcher and to gen actual root node we need to take it's parent.
        if (auto any_type = dynamic_pointer_cast<pattern::op::AnyOutput>(root))
        {
            root = any_type->input_value(0).get_node_shared_ptr();
        }

        // if root is an operation from opset or has pattern::op::WrapType type then we can extract
        // it's type
        // and use it in unordered_map as key for fast MatcherPass search. Otherwise type is unknown
        // and default algorithm is used.
        if (auto p = dynamic_pointer_cast<pattern::op::Pattern>(root))
        {
            if (auto any_type = dynamic_pointer_cast<pattern::op::WrapType>(p))
            {
                for (const auto& root_type_info : any_type->get_wrapped_types())
                {
                    type_to_matcher[root_type_info].push_back(matcher_index);
                }
            }
            else
            {
                all_roots_has_type = false;
                break;
            }
        }
        else
        {
            type_to_matcher[root->get_type_info()].push_back(matcher_index);
        }

        // TODO: traverse parents for root_type_info in order to register complete list of matchers
        // including ones triggered by parent type info.
    }

    // This lambda preforms execution of particular MatcherPass on given node.
    // It automatically handles nodes registered by MatcherPass during transformation and set
    // transformation callback.
    auto run_matcher_pass = [&](std::shared_ptr<MatcherPass> m_pass,
                                std::shared_ptr<Node> node) -> bool {
        // Keep this property check for backward compatibility. In future transformation property
        // will be deprecated and removed.
        if (m_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && f->is_dynamic())
        {
            NGRAPH_DEBUG << "matcher callback requires static shape but the "
                            "function is dynamic, skipping this "
                            "optimization till the shapes are fully "
                            "materialized";
            return false;
        }

        // Apply MatcherPass. In case if it returns true no other MatcherPasses will apply
        // to this node
        bool status = m_pass->apply(node);

        // In case if MatcherPass registered nodes they will be added to the beginning of execution
        // queue
        const auto& new_nodes = m_pass->get_new_nodes();
        if (!new_nodes.empty())
        {
            // Need to push nodes in reverse order as we expect that nodes in new_nodes
            // vector are in topological order
            for (auto it = new_nodes.rbegin(); it != new_nodes.rend(); it++)
            {
                nodes_to_run.emplace_front(*it);
            }
            m_pass->clear_new_nodes();
        }
        return status;
    };

    // list of matchers to run for a node; define here to keep memory allocated
    std::vector<size_t> matcher_passes_to_run;

    while (!nodes_to_run.empty())
    {
        auto node = nodes_to_run.front();
        nodes_to_run.pop_front();
        // Recursive apply Matchers for sub-graph based nodes
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
        {
            if (auto sub_graph = sub_graph_node->get_function())
            {
                run_on_function(sub_graph);
            }
        }
        // Temporary keep this GraphRewrite property for backward compatibility
        if (m_enable_shape_inference)
        {
            node->revalidate_and_infer_types();
        }
        // If all Matchers in MatcherPasses has type based root node then we apply efficient
        // algorithm for finding matchers
        if (all_roots_has_type)
        {
            const DiscreteTypeInfo* node_type_info = &node->get_type_info();
            matcher_passes_to_run.clear();
            while (node_type_info)
            {
                auto matchers = type_to_matcher.find(*node_type_info);
                if (matchers != type_to_matcher.end())
                {
                    // do not run found matchers immediately, need to collect all matchers for
                    // parents
                    // and sort them in order of the registration
                    matcher_passes_to_run.insert(matcher_passes_to_run.end(),
                                                 matchers->second.begin(),
                                                 matchers->second.end());
                }
                node_type_info = node_type_info->parent;
            }

            std::sort(matcher_passes_to_run.begin(), matcher_passes_to_run.end());

            // TODO: type_to_matcher with just collected list of matchers to enable
            // fast processing at the next time when node with the same type will be processed

            for (size_t matcher_index : matcher_passes_to_run)
            {
                if (run_matcher_pass(m_matchers[matcher_index], node))
                {
                    rewritten = true;
                    break;
                }
            }
        }
        // Otherwise we use default algorithm that iterates over all registered matcher passes
        else
        {
            for (auto& m_pass : m_matchers)
            {
                // Skip passes that are disabled
                if (pass_config->is_disabled(m_pass->get_type_info()))
                    continue;

                if (run_matcher_pass(m_pass, node))
                {
                    rewritten = true;
                    break;
                }
            }
        }
    }
    return rewritten;
}

void pass::GraphRewrite::add_matcher(const shared_ptr<pattern::Matcher>& m,
                                     const graph_rewrite_callback& callback,
                                     const PassPropertyMask& property)
{
    m_matchers.push_back(std::make_shared<MatcherPass>(
        m->get_name(),
        m,
        [m, callback](const std::shared_ptr<Node>& node) -> bool {
            NGRAPH_DEBUG << "Running matcher " << m->get_name() << " on " << node;
            if (m->match(node->output(0)))
            {
                NGRAPH_DEBUG << "Matcher " << m->get_name() << " matched " << node;
                NGRAPH_PASS_CALLBACK(m);
                bool status = callback(*m.get());
                // explicitly clear Matcher state because it holds pointers to matched nodes
                m->clear_state();
                return status;
            }
            m->clear_state();
            return false;
        },
        property));
}

void pass::GraphRewrite::add_matcher(const shared_ptr<pattern::Matcher>& m,
                                     const graph_rewrite_callback& callback)
{
    NGRAPH_SUPPRESS_DEPRECATED_START
    // TODO: before deprecate this function, by default expect the
    // callback require static shape.
    add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
    NGRAPH_SUPPRESS_DEPRECATED_END
}

void pass::GraphRewrite::set_pass_config(const std::shared_ptr<PassConfig>& rhs)
{
    auto pass_config = get_pass_config();
    // We have to preserve disabled passes because in case when we register matchers inside
    // GraphRewrite c-tor we work with local PassConfig instance.
    // For example:
    //
    // class ExampleGraphRewrite: public pass::GraphRewrite {
    //      ExampleGraphRewrite() {
    //          add_mather<TestMatcher1, false /* disabled by default */>();
    //          add_mather<TestMatcher2>();
    //      }
    // };
    //
    // When we call add_matcher inside c-tor we automatically work with locally created PassConfig
    // instance that is not shared. So when instance of this pass is being created in pass::Manager
    // we set shared PassConfig but we will override already existing rules inside local config. To
    // resolve this we have to copy disabled passes from local PassConfig to shared but we take into
    // account that if passes were manually enabled we do not add them.
    rhs->add_disabled_passes(*pass_config);
    PassBase::set_pass_config(rhs);

    // update nested transformations with new shared pass_config
    for (auto& pass : m_matchers)
    {
        pass->set_pass_config(rhs);
    }
}

void pass::RecurrentGraphRewrite::add_matcher(
    const std::shared_ptr<pattern::RecurrentMatcher>& m,
    const ngraph::recurrent_graph_rewrite_callback& callback,
    const PassPropertyMask& property)
{
    m_matchers.push_back(std::make_shared<MatcherPass>(
        "Recurrent matcher",
        nullptr,
        [m, callback](const std::shared_ptr<Node>& node) {
            NGRAPH_DEBUG << "Running recurrent matcher on " << node;
            if (m->match(node->output(0)))
            {
                NGRAPH_DEBUG << "Recurrent matcher matched " << m.get();
                return callback(*m.get());
            }
            return false;
        },
        property));
}

void pass::RecurrentGraphRewrite::add_matcher(
    const std::shared_ptr<pattern::RecurrentMatcher>& m,
    const ngraph::recurrent_graph_rewrite_callback& callback)
{
    // TODO: before deprecate this function, by default expect the
    // callback require static shape.
    add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
}

bool pass::RecurrentGraphRewrite::run_on_function(shared_ptr<Function> f)
{
    bool changed = false;
    size_t i = 0;

    // This check is very expensive and is only needed for experimental features, so we will hide
    // it behind an environment variable for now. TODO: Find a less expensive way to handle this.
    static bool s_rerun_dynamic_check = getenv_bool("NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK");

    auto run_matchers = [&]() -> bool {
        bool is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
        for (auto node : f->get_ops())
        {
            for (auto& m_pass : m_matchers)
            {
                if (is_dyn_func && m_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE))
                {
                    NGRAPH_DEBUG << "matcher callback requires static shape but the "
                                    "function is dynamic, skipping this "
                                    "optimization till the shapes are fully "
                                    "materialized";
                    continue;
                }
                if (m_pass->apply(node))
                {
                    // If call back may change function's is_dynamic state, we need to
                    // update the cached value.
                    if (m_pass->get_property(PassProperty::CHANGE_DYNAMIC_STATE))
                    {
                        is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
                    }
                    return true;
                }
            }
        }
        return false;
    };

    do
    {
        changed = run_matchers();
        i++;
    } while (changed && i < m_num_iters);
    return changed;
}

void ngraph::pass::MatcherPass::register_matcher(const std::shared_ptr<ngraph::pattern::Matcher>& m,
                                                 const ngraph::graph_rewrite_callback& callback,
                                                 const PassPropertyMask& property)
{
    set_name(m->get_name());
    set_property(property, true);
    m_matcher = m;
    m_handler = [m, callback](const std::shared_ptr<Node>& node) -> bool {
        if (m->match(node->output(0)))
        {
            NGRAPH_DEBUG << "Matcher " << m->get_name() << " matched " << node;
            NGRAPH_PASS_CALLBACK(m);
            bool status = callback(*m.get());
            // explicitly clear Matcher state because it holds pointers to matched nodes
            m->clear_state();
            return status;
        }
        m->clear_state();
        return false;
    };
}

bool ngraph::pass::MatcherPass::apply(std::shared_ptr<ngraph::Node> node)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph,
                       pass::internal::perf_counters_graph_rewrite()[get_type_info()]);
    m_new_nodes.clear();
    if (m_handler)
        return m_handler(node);
    return false;
}
