// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/manager.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "itt.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"
#include "openvino/util/env_util.hpp"
#include "perf_counters.hpp"

#include <assert.h>

#define EMUTEX_TRACE_ENABLED 1

using namespace std;

namespace ov {
namespace pass {
namespace {
PerfCounters& perf_counters() {
    static PerfCounters counters;
    return counters;
}
}  // namespace

#ifdef EMUTEX_TRACE_ENABLED
using Graph = std::unordered_map<std::string, std::unordered_set<std::string>>;

template <typename GraphFunctionT>
class FunctionTracer
{
public:
    FunctionTracer(std::function<void (const std::string& pass_name, std::shared_ptr<ngraph::Function>)> OnNextPassReturnHook, GraphFunctionT);
    void OnNextPassReturn(const std::string& pass_name, GraphFunctionT);
private:
    std::function<void (const std::string& pass_name, std::shared_ptr<ngraph::Function>)> m_OnNextPassReturnHook;
    Graph m_graph;
};

namespace {

//---------------------------------------------------------------------------------------

using Graph = std::unordered_map<std::string, std::unordered_set<std::string>>;

Graph BuildGraph(std::shared_ptr<ngraph::Function> function)
{
    Graph graph;

    for (const shared_ptr<Node>& node : function->get_ops())
    {
        std::string node_name = node->get_name();

        for (auto output : node->outputs())
        {
            for (auto input : output.get_target_inputs())
            {
                std::string child_node_name = input.get_node()->get_name();
                graph[node_name].insert(child_node_name);
                graph[child_node_name].insert(node_name);
            }
        }
    }

    return graph;
}

//---------------------------------------------------------------------------------------

/**
 * @brief find layers that exist in dst_graph but don't exist in src_graph
 * 
 * @param src_graph 
 * @param dst_graph 
 * @return std::vector<std::string> 
 */
std::vector<std::string> FindNewLayers(const Graph & src_graph, const Graph & dst_graph)
{
    std::vector<std::string> found_layers;

    for (auto graph_it : dst_graph)
    {
        const std::string & layer_name = graph_it.first;
        if (src_graph.find(layer_name) == src_graph.end())
            found_layers.push_back(layer_name);
    }

    return found_layers;
}

//---------------------------------------------------------------------------------------


} // namespace

template <typename GraphFunctionT>
FunctionTracer<GraphFunctionT>::FunctionTracer(std::function<void (const std::string& pass_name, std::shared_ptr<ngraph::Function>)> OnNextPassReturnHook, GraphFunctionT function)
    : m_OnNextPassReturnHook(OnNextPassReturnHook),
      m_graph(BuildGraph(function))
{
}

template <typename GraphFunctionT>
void FunctionTracer<GraphFunctionT>::OnNextPassReturn(const std::string& pass_name, GraphFunctionT function)
{
    Graph new_graph = BuildGraph(function);

    std::vector<std::string> add_layers_list = FindNewLayers(m_graph, new_graph);

    for (const std::string& name: add_layers_list)
    {
        std::cout << "EMUTEX DEBUG ngraph [" << pass_name << "] add layer " << name << std::endl;
    }

    std::vector<std::string> removed_layers_list = FindNewLayers(new_graph, m_graph);

    for (const std::string& name: removed_layers_list)
    {
        std::cout << "EMUTEX DEBUG ngraph [" << pass_name << "] remove layer " << name << std::endl;
    }

    m_graph = new_graph;

    if (m_OnNextPassReturnHook)
        m_OnNextPassReturnHook(pass_name, function);
}
#endif // EMUTEX_TRACE_ENABLED
//---------------------------------------------------------------------------------------

}  // namespace pass
}  // namespace ov

ov::pass::Manager::Manager()
    : m_pass_config(std::make_shared<PassConfig>()),
      m_visualize(ov::util::getenv_bool("NGRAPH_ENABLE_VISUALIZE_TRACING") ||
                  ov::util::getenv_bool("OV_ENABLE_VISUALIZE_TRACING")) {}

ov::pass::Manager::~Manager() = default;

ov::pass::Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config) : m_pass_config(std::move(pass_config)) {}

void ov::pass::Manager::set_per_pass_validation(bool new_state) {
    m_per_pass_validation = new_state;
}

#define EMUTEX_DEBUG_CHECKPOINT std::cout << __FILE__ << ":" << __LINE__ << std::endl;

void ov::pass::Manager::run_passes(shared_ptr<ov::Model> func) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    OV_ITT_SCOPED_TASK(ov::itt::domains::nGraph, "pass::Manager::run_passes");

    static bool profile_enabled =
        ov::util::getenv_bool("NGRAPH_PROFILE_PASS_ENABLE") || ov::util::getenv_bool("OV_PROFILE_PASS_ENABLE");

#ifdef EMUTEX_TRACE_ENABLED
    FunctionTracer<shared_ptr<ngraph::Function>> func_tracer(m_OnNextPassReturnHook, func);
#endif

    size_t index = 0;
    ngraph::stopwatch pass_timer;
    ngraph::stopwatch overall_timer;
    overall_timer.start();
    bool function_changed = false;
    for (auto& pass : m_pass_list) {
        if (m_pass_config->is_disabled(pass->get_type_info())) {
            NGRAPH_DEBUG << "Pass " << pass->get_name() << " is disabled";
            continue;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::nGraphPass_LT, pass::perf_counters()[pass->get_type_info()]);

        pass_timer.start();

        if (auto matcher_pass = dynamic_pointer_cast<MatcherPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (matcher_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic()) {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "model is dynamic. Skipping this transformation";
                continue;
            }
            // GraphRewrite is a temporary container for MatcherPass to make execution
            // on on entire ngraph::Function
            function_changed = GraphRewrite(matcher_pass).run_on_model(func);
#ifdef EMUTEX_TRACE_ENABLED
            func_tracer.OnNextPassReturn(pass->get_name(), func);
#endif
        } else if (auto function_pass = dynamic_pointer_cast<ModelPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (function_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic()) {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "model is dynamic. Skipping this transformation";
                continue;
            }

            if (dynamic_pointer_cast<Validate>(pass)) {
                if (function_changed) {
                    function_pass->run_on_model(func);
                    function_changed = false;
#ifdef EMUTEX_TRACE_ENABLED
                    func_tracer.OnNextPassReturn(pass->get_name(), func);
#endif
                }
            } else {
                function_changed = function_pass->run_on_model(func);
#ifdef EMUTEX_TRACE_ENABLED
                    func_tracer.OnNextPassReturn(pass->get_name(), func);
#endif
            }
        } else if (auto node_pass = dynamic_pointer_cast<ngraph::pass::NodePass>(pass)) {
            if (node_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic()) {
                NGRAPH_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                             << "model is dynamic. Skipping this transformation";
                continue;
            }
            for (const shared_ptr<Node>& n : func->get_ops()) {
                function_changed |= node_pass->run_on_node(n);
#ifdef EMUTEX_TRACE_ENABLED
            func_tracer.OnNextPassReturn(pass->get_name(), func);
#endif
            }
#ifdef EMUTEX_TRACE_ENABLED
            func_tracer.OnNextPassReturn(pass->get_name(), func);
#endif
        }

        if (m_visualize) {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = func->get_name() + std::string("_") + index_str + std::string("_") + pass->get_name();

            if (m_visualize) {
                auto file_ext = "svg";
                pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.run_on_model(func);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled) {
            cout << setw(7) << pass_timer.get_milliseconds() << "ms " << pass->get_name() << "\n";
        }
    }
    if (profile_enabled) {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
