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

#include <algorithm>
#include <iostream>
#include <regex>
#include <unordered_set>
#include <vector>

#include "graph_rewrite.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

// GraphRewrite algorithm:
// GraphRewrite processes an input graph in an topological order(i.e. args before users)
// Given the following graph:          Abs2
//                                   /       \
//                         Constant1         Add4 - Result5
//                                   \      /
//                                    Neg3
//
// The topological order would be : `Constant1`, `Abs2`, `Neg3`, `Add4`, `Result5`
// Note, `Abs2` comes before `Neg3` as `Abs2`'s id = 2 is *less* than `Neg3`'s one (id = 3)
// Next, GraphRewrite will invoke matchers registered in an order registered in a c-tor
// i.e. if a c-tor calls `construct_m1()`; `construct_m2()`; `construct_m3()`;
// Matchers will be called as follows: `m1`, `m2`, `m3`
// Matchers should only replace nodes in the graph that come before the current root
// node in the topological order. For example, if Matcher matches Neg3, it should only
// replace nodes `Abs2` and `Constant1` if needed
// This gives Matchers a nice cascading property. For example, if m1 folds `Abs2(Constant1)`
// and `m2` folds `Neg3(Constant1)` when `m3` is called on `Add4` it will discover that
// both `Abs2` and `Neg3` were already replaced by constants, so `Add4` will also be folded into
// one.
// If any Matcher succeeds the rest of the matchers will **not** be called.
// E.g. if `m1` succeeds and replaces `Abs2` with a new constant, nor `m2` or `m3` will be called
// However, sometimes, you will need more than one fusion occur on the same node.
// In this case, you should be able to request another pass of GraphRewrite.
// To request another pass, you will need to register fusions in a callback:
// i.e. you will need to pass `this` into a callback and then call `this->construct_X`
// This will schedule another pass of GraphRewrite with the following fusion.
// This approach should only be used if you are either:
// a) need more than one fusion occur on the same node
// b) you are modifying nodes after the current node in the topological order
// c) there's no linear order of fusions which will give
//    the correct final fusion. i.e. the same fusion needs to occur before and after some other
//    fusion

bool pass::GraphRewrite::run_on_function(shared_ptr<Function> f)
{
    bool rewritten = false;
    const size_t NUM_TRIES = 10;
    size_t tries = NUM_TRIES;
    vector<MatchClosure> original_matchers{m_matchers};
    // This check is very expensive and is only needed for experimental features, so we will hide
    // it behind an environment variable for now. TODO: Find a less expensive way to handle this.
    static bool s_rerun_dynamic_check = getenv_bool("NGRAPH_GRAPH_REWRITE_RERUN_DYNAMIC_CHECK");
    bool is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
    do
    {
        rewritten = false;
        // m_matchers may contain newly constructed matchers for matchers
        // that need multiple passes. See comments above.
        vector<MatchClosure> matchers_to_run{m_matchers};
        m_matchers.clear();
        for (auto node : f->get_ordered_ops())
        {
            if (m_enable_shape_inference)
            {
                node->revalidate_and_infer_types();
            }
            for (auto& closure : matchers_to_run)
            {
                if (is_dyn_func && closure.property[PassProperty::REQUIRE_STATIC_SHAPE])
                {
                    NGRAPH_DEBUG << "matcher callback requires static shape but the "
                                    "function is dynamic, skipping this "
                                    "optimization till the shapes are fully "
                                    "materialized";
                    continue;
                }
                if (closure.handler(node))
                {
                    rewritten = true;
                    // If call back may change function's is_dynamic state, we need to
                    // update the cached value.
                    if (closure.property.is_set(PassProperty::CHANGE_DYNAMIC_STATE))
                    {
                        is_dyn_func = s_rerun_dynamic_check && f->is_dynamic();
                    }
                    break;
                }
            }
        }

    } while (rewritten && m_matchers.size() > 0 && tries--);

    m_matchers.assign(original_matchers.begin(), original_matchers.end());
    return (NUM_TRIES - tries) > 1; // this means a graph was transformed
}

static vector<regex> initialize_fusion_regexes()
{
    static const string nsf = getenv_string("NGRAPH_DISABLED_FUSIONS");
    vector<regex> regexes;
    if (!nsf.empty())
    {
        const auto sregexes = split(nsf, ';');

        transform(sregexes.begin(),
                  sregexes.end(),
                  back_inserter(regexes),
                  [](const string& c) -> regex { return regex(c); });
    }
    return regexes;
}

bool pass::GraphRewriteBase::is_enabled(const std::string& name) const
{
    // note, regexes are static to avoid re-initialization
    static const auto regexes = initialize_fusion_regexes();

    for (const auto& regex : regexes)
    {
        if (regex_match(name, regex))
        {
            NGRAPH_DEBUG << "Disabling matcher " << name;
            return false;
        }
    }

    return true;
}

void pass::GraphRewriteBase::add_handler(const std::string& name,
                                         function<bool(const std::shared_ptr<Node>&)> handler,
                                         const PassPropertyMask& property)
{
    if (is_enabled(name))
    {
        m_matchers.push_back({name, handler, property});
        // If any matcher call back may change dynamic state, we need to
        // update the pass property.
        if (property.is_set(PassProperty::CHANGE_DYNAMIC_STATE))
        {
            set_property(PassProperty::CHANGE_DYNAMIC_STATE, true);
        }
    }
}

void pass::GraphRewrite::add_matcher(const shared_ptr<pattern::Matcher>& m,
                                     const graph_rewrite_callback& callback,
                                     const PassPropertyMask& property)
{
    add_handler(m->get_name(),
                [m, callback](const std::shared_ptr<Node>& node) -> bool {
                    NGRAPH_DEBUG << "Running matcher " << m->get_name() << " on " << node;
                    if (m->match(node->output(0)))
                    {
                        NGRAPH_DEBUG << "Matcher " << m->get_name() << " matched " << node;
                        return callback(*m.get());
                    }
                    return false;
                },
                property);
}

void pass::GraphRewrite::add_matcher(const shared_ptr<pattern::Matcher>& m,
                                     const graph_rewrite_callback& callback)
{
    // TODO: before deprecate this function, by default expect the
    // callback require static shape.
    add_matcher(m, callback, {PassProperty::REQUIRE_STATIC_SHAPE});
}

void pass::GraphRewrite::copy_matchers(const std::shared_ptr<GraphRewrite>& pass)
{
    for (auto& m : pass->m_matchers)
    {
        m_matchers.push_back(m);
    }
}

void pass::RecurrentGraphRewrite::add_matcher(
    const std::shared_ptr<pattern::RecurrentMatcher>& m,
    const ngraph::recurrent_graph_rewrite_callback& callback,
    const PassPropertyMask& property)
{
    add_handler("Reurrent matcher",
                [m, callback](const std::shared_ptr<Node>& node) {
                    NGRAPH_DEBUG << "Running recurrent matcher on " << node;
                    if (m->match(node->output(0)))
                    {
                        NGRAPH_DEBUG << "Recurrent matcher matched " << m.get();
                        return callback(*m.get());
                    }
                    return false;
                },
                property);
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
            for (auto& closure : m_matchers)
            {
                if (is_dyn_func && closure.property[PassProperty::REQUIRE_STATIC_SHAPE])
                {
                    NGRAPH_DEBUG << "matcher callback requires static shape but the "
                                    "function is dynamic, skipping this "
                                    "optimization till the shapes are fully "
                                    "materialized";
                    continue;
                }
                if (closure.handler(node))
                {
                    // If call back may change function's is_dynamic state, we need to
                    // update the cached value.
                    if (closure.property.is_set(PassProperty::CHANGE_DYNAMIC_STATE))
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
