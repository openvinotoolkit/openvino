// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "itt.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"
#include "perf_counters.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace pass
    {
        namespace internal
        {
            PerfCounters& perf_counters()
            {
                static PerfCounters counters;
                return counters;
            }
        } // namespace internal
    }     // namespace pass
} // namespace ngraph

pass::Manager::Manager()
    : m_pass_config(std::make_shared<PassConfig>())
    , m_visualize(getenv_bool("NGRAPH_ENABLE_VISUALIZE_TRACING"))
{
}

pass::Manager::~Manager() {}

pass::Manager::Manager(std::shared_ptr<ngraph::pass::PassConfig> pass_config)
    : m_pass_config(std::move(pass_config))
{
}

void pass::Manager::run_passes(shared_ptr<Function> func)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "pass::Manager::run_passes");

    static bool profile_enabled = getenv_bool("NGRAPH_PROFILE_PASS_ENABLE");

    size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    bool function_changed = false;
    for (auto& pass : m_pass_list)
    {
        if (m_pass_config->is_disabled(pass->get_type_info()))
        {
            NGRAPH_DEBUG << "Pass " << pass->get_name() << " is disabled";
            continue;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE,
                     itt::domains::nGraphPass_LT,
                     pass::internal::perf_counters()[pass->get_type_info()]);

        pass_timer.start();

        NGRAPH_SUPPRESS_DEPRECATED_START
        if (auto matcher_pass = dynamic_pointer_cast<MatcherPass>(pass))
        {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (matcher_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }
            // GraphRewrite is a temporary container for MatcherPass to make execution
            // on on entire ngraph::Function
            function_changed = GraphRewrite(matcher_pass).run_on_function(func);
        }
        else if (auto function_pass = dynamic_pointer_cast<FunctionPass>(pass))
        {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (function_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }

            if (dynamic_pointer_cast<Validate>(pass))
            {
                if (function_changed)
                {
                    function_pass->run_on_function(func);
                    function_changed = false;
                }
            }
            else
            {
                function_changed = function_pass->run_on_function(func);
            }
        }
        else if (auto node_pass = dynamic_pointer_cast<NodePass>(pass))
        {
            if (node_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic())
            {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "function is dynamic. Skipping this transformation";
                continue;
            }
            for (shared_ptr<Node> n : func->get_ops())
            {
                function_changed |= node_pass->run_on_node(n);
            }
        }
        NGRAPH_SUPPRESS_DEPRECATED_END

        if (m_visualize)
        {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = func->get_name() + std::string("_") + index_str +
                                 std::string("_") + pass->get_name();

            if (m_visualize)
            {
                static const string format = getenv_string("NGRAPH_VISUALIZE_TRACING_FORMAT");
                auto file_ext = format.empty() ? "svg" : format;
                pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.run_on_function(func);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled)
        {
            cout << setw(7) << pass_timer.get_milliseconds() << "ms " << pass->get_name() << "\n";
        }
    }
    if (profile_enabled)
    {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
}
