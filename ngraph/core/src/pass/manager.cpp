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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

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

using namespace std;
using namespace ngraph;

pass::Manager::Manager()
    : m_visualize(getenv_bool("NGRAPH_ENABLE_VISUALIZE_TRACING"))
    , m_statistics(getenv_bool("NGRAPH_ENABLE_STATISTICS_TRACING"))
{
}

pass::Manager::~Manager()
{
}

struct OperationDescription
{
    // All non dynamic dimesions are all set to 1
    typedef PartialShape DynamicMask;

    std::string name;
    int64_t version;
    std::vector<DynamicMask> inputs;
    std::vector<DynamicMask> outputs;

    explicit OperationDescription(std::shared_ptr<Node> node)
    {
        name = node->get_type_info().name;
        version = node->get_type_info().version;
        create_mask(node->inputs(), inputs);
        create_mask(node->outputs(), outputs);
    }

    OperationDescription(const std::string _name,
                         int64_t _version,
                         std::vector<DynamicMask> _inputs,
                         std::vector<DynamicMask> _outputs)
        : name(_name)
        , version(_version)
        , inputs(_inputs)
        , outputs(_outputs)
    {
    }

    template <typename T>
    void create_mask(const std::vector<T>& ports, std::vector<DynamicMask>& dest)
    {
        dest.clear();
        for (const auto& port : ports)
        {
            PartialShape shape = port.get_partial_shape();
            if (!shape.rank().is_dynamic())
            {
                size_t rank = shape.rank().get_length();
                for (size_t i = 0; i < rank; ++i)
                {
                    if (!shape[i].is_dynamic())
                    {
                        shape[i] = 1; // set all not dynamic dimensions to 1
                    }
                }
            }
            dest.push_back(shape);
        }
    }

    static bool less(const std::vector<OperationDescription::DynamicMask>& x1,
                     const std::vector<OperationDescription::DynamicMask>& x2)
    {
        if (x1.size() < x2.size())
            return true;

        if (x2.size() < x1.size())
            return false;

        for (size_t i = 0; i < x1.size(); ++i)
        {
            const auto& shape1 = x1[i];
            const auto& shape2 = x2[i];

            if (shape1.rank().is_dynamic() && shape2.rank().is_static())
                return true;

            if (shape1.rank().is_static() && shape2.rank().is_dynamic())
                return false;

            if (shape1.rank().is_static() && shape2.rank().is_static())
            {
                if (shape1.rank().get_length() < shape2.rank().get_length())
                    return true;

                if (shape2.rank().get_length() < shape1.rank().get_length())
                    return false;

                for (size_t j = 0; j < shape1.rank().get_length(); ++j)
                {
                    if (shape1[j].is_dynamic() && !shape2[j].is_dynamic())
                        return true;

                    if (!shape1[j].is_dynamic() && shape2[j].is_dynamic())
                        return false;
                    // if they both are dynamic or static, then they are equal
                }
            }
        }

        return false;
    }

    bool is_dynamic() const
    {
        auto has_dynamic = [](const PartialShape& shape) { return shape.is_dynamic(); };
        return inputs.end() != std::find_if(inputs.begin(), inputs.end(), has_dynamic) ||
               outputs.end() != std::find_if(outputs.begin(), outputs.end(), has_dynamic);
    }
};

bool operator<(const OperationDescription& x1, const OperationDescription& x2)
{
    if (x1.name < x2.name)
        return true;

    if (x2.name < x1.name)
        return false;

    if (x1.version < x2.version)
        return true;

    if (x2.version < x1.version)
        return false;

    if (OperationDescription::less(x1.inputs, x2.inputs))
        return true;
    else if (OperationDescription::less(x2.inputs, x1.inputs))
        return false;

    if (OperationDescription::less(x1.outputs, x2.outputs))
        return true;
    else if (OperationDescription::less(x2.outputs, x1.outputs))
        return false;

    return false;
}

void print_partial_shape(std::ostream& out, const PartialShape& shape)
{
    if (shape.rank().is_dynamic())
        out << '?';
    else
    {
        out << '{';
        size_t rank = shape.rank().get_length();
        for (size_t i = 0; i < rank; ++i)
        {
            if (i > 0)
                out << ", ";
            if (shape[i].is_static())
                out << 'S';
            else
                out << shape[i];
        }
        out << '}';
    }
}

std::ostream& operator<<(std::ostream& out, const std::vector<PartialShape>& shapes)
{
    out << "(";
    for (size_t i = 0; i < shapes.size(); ++i)
    {
        if (i > 0)
            out << ", ";
        print_partial_shape(out, shapes[i]);
    }
    out << ")";
    return out;
}

std::ostream& operator<<(std::ostream& out, const OperationDescription& x)
{
    out << x.name << "-" << x.version << x.inputs << " --> " << x.outputs;
    return out;
}

void pass::Manager::run_passes(shared_ptr<Function> func)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "pass::Manager::run_passes");

    static bool profile_enabled = getenv_bool("NGRAPH_PROFILE_PASS_ENABLE");

    static size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    bool function_changed = false;
    for (auto& pass : m_pass_list)
    {
        pass_timer.start();
        if (!m_has_default_callback)
        {
            pass->set_callback(m_transformation_callback);
        }

        try
        {
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
                if (node_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) &&
                    func->is_dynamic())
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
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception std::exception thrown while executing transformation: "
                      << pass->get_name() << "\n";
            std::cerr << "Exception message: " << e.what() << "\n";
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown exception thrown while executing transformation: "
                      << pass->get_name() << "\n";
            throw;
        }

        if (m_visualize || m_statistics)
        {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 6;
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

            if (m_statistics)
            {
                // auto x1 = OperationDescription("name", 0, {PartialShape{Dimension(), 1}}, {});
                // auto x2 = OperationDescription("name", 0, {PartialShape{Dimension(),
                // Dimension()}}, {});
                // bool b1 = x1 < x2;
                // bool b2 = x2 < x1;

                // std::ofstream allops(base_filename + ".all.dynops.txt");
                std::set<OperationDescription> operations;
                for (auto& node : func->get_ops())
                {
                    OperationDescription od(node);
                    // allops << od << '\n';
                    if (od.is_dynamic())
                        operations.insert(od);
                }
                std::ofstream list(base_filename + ".dynops.txt");
                for (const auto& x : operations)
                {
                    list << x << std::endl;
                }
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
