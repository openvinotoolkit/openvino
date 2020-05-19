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
#ifdef _WIN32
#else
#include <cxxabi.h>
#endif
#include <iomanip>
#include <iostream>
#include <memory>

#include "ngraph/env_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/serialize.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

pass::Manager::Manager()
    : m_visualize(getenv_bool("NGRAPH_ENABLE_VISUALIZE_TRACING"))
    , m_serialize(getenv_bool("NGRAPH_ENABLE_SERIALIZE_TRACING"))

{
}

pass::Manager::~Manager()
{
}

void pass::Manager::run_passes(shared_ptr<Function> func, bool /* transitive */)
{
    static bool profile_enabled = getenv_bool("NGRAPH_PROFILE_PASS_ENABLE");

    get_state().set_function(func);
    vector<std::pair<shared_ptr<Function>, bool>> fs{std::make_pair(func, func->is_dynamic())};
    vector<shared_ptr<Function>> f_array{func};

    size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    for (shared_ptr<PassBase> pass : m_pass_list)
    {
        pass_timer.start();
        pass->set_state(get_state());
        auto module_pass = dynamic_pointer_cast<ModulePass>(pass);
        auto function_pass = dynamic_pointer_cast<FunctionPass>(pass);
        auto node_pass = dynamic_pointer_cast<NodePass>(pass);
        auto call_graph_pass = dynamic_pointer_cast<CallGraphPass>(pass);
        if (module_pass)
        {
            if (auto vt_pass = dynamic_pointer_cast<pass::VisualizeTree>(module_pass))
            {
                vt_pass->set_ops_to_details(get_state().get_visualize_tree_ops_map());
            }
            module_pass->run_on_module(f_array);
        }
        else if (function_pass)
        {
            for (auto f_pair : fs)
            {
                shared_ptr<Function> f = f_pair.first;
                // This checks is to skip the graph optimization when the graph pass relies on
                // static shape but the function state is dynamic.
                // we update the function dynamic state only if we run the graph pass successfully.
                if (function_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                    f_pair.second)
                {
                    continue;
                }
                bool function_modified = function_pass->run_on_function(f);
                // If the pass may change the function's is_dynamic property, we need to
                // update the cached value.
                if (function_modified &&
                    function_pass->get_property(PassProperty::CHANGE_DYNAMIC_STATE))
                {
                    f_pair.second = f->is_dynamic();
                }
            }
        }
        else if (node_pass)
        {
            for (auto f_pair : fs)
            {
                shared_ptr<Function> f = f_pair.first;
                if (node_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && f_pair.second)
                {
                    continue;
                }
                for (shared_ptr<Node> n : f->get_ops())
                {
                    node_pass->run_on_node(n);
                }
            }
        }
        else if (call_graph_pass)
        {
            for (auto f_pair : fs)
            {
                shared_ptr<Function> f = f_pair.first;
                if (call_graph_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                    f_pair.second)
                {
                    continue;
                }
                bool function_modified = call_graph_pass->run_on_call_graph(f->get_ordered_ops());
                f_pair.second = (function_modified == true) ? f->is_dynamic() : f_pair.second;
            }
        }

        if (m_visualize || m_serialize)
        {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = f_array.at(0)->get_name() + std::string("_") + index_str +
                                 std::string("_") + m_pass_names.at(index);

            if (m_visualize)
            {
                static const string format = getenv_string("NGRAPH_VISUALIZE_TRACING_FORMAT");
                auto file_ext = format.empty() ? "svg" : format;
                pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.set_ops_to_details(get_state().get_visualize_tree_ops_map());
                vt.run_on_module(f_array);
            }

            if (m_serialize)
            {
                pass::Serialization st(base_filename + ".json");
                st.run_on_module(f_array);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled)
        {
            PassBase* p = pass.get();
            string name = typeid(*p).name();
#ifndef _WIN32
            int status;
            name = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
#endif
            cout << setw(7) << pass_timer.get_milliseconds() << "ms " << name << "\n";
        }
    }
    if (profile_enabled)
    {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
}

pass::ManagerState& pass::Manager::get_state()
{
    return m_state;
}
