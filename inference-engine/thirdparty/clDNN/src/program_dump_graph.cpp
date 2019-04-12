/*
// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_dump_graph.h"
#include "to_string_utils.h"
#include "data_inst.h"
#include "condition_inst.h"

#include "gpu/ocl_toolkit.h"

#include "to_string_utils.h"

#include <algorithm>
#include <vector>

namespace cldnn
{
    namespace
    {
        static const std::vector<std::string> colors =
        {
            "chartreuse",
            "aquamarine",
            "gold",
            "green",
            "blue",
            "cyan",
            "azure",
            "beige",
            "bisque",
            "blanchedalmond",
            "blueviolet",
            "brown",
            "burlywood",
            "cadetblue",
            "chocolate",
            "coral",
            "cornflowerblue",
            "cornsilk",
            "crimson",
            "aliceblue",
            "antiquewhite",
            "deeppink",
            "deepskyblue",
            "dimgray",
            "dodgerblue",
            "firebrick",
            "floralwhite",
            "forestgreen",
            "gainsboro",
            "ghostwhite",
            "goldenrod",
            "greenyellow",
            "honeydew",
            "hotpink",
            "indianred",
            "indigo",
            "ivory",
            "khaki",
            "lavender",
            "lavenderblush",
            "lawngreen",
            "lemonchiffon",
            "lightblue",
            "lightcoral",
            "lightcyan",
            "lightgoldenrodyellow",
            "lightgray",
            "lightgrey",
            "lightpink",
            "lightsalmon",
            "lightseagreen",
            "lightskyblue",
            "lightsteelblue",
            "lightyellow",
            "lime",
            "limegreen",
            "linen",
            "magenta",
            "maroon",
            "mediumaquamarine",
            "mediumblue",
            "mediumorchid",
            "mediumpurple",
            "mediumseagreen",
            "mediumslateblue",
            "mediumspringgreen",
            "mediumturquoise",
            "mediumvioletred",
            "midnightblue",
            "mintcream",
            "mistyrose",
            "moccasin",
            "navajowhite",
            "navy",
            "oldlace",
            "olive",
            "olivedrab",
            "orange",
            "orangered",
            "orchid",
            "palegoldenrod",
            "palegreen",
            "paleturquoise",
            "palevioletred",
            "papayawhip",
            "peachpuff",
            "peru",
            "pink",
            "plum",
            "powderblue",
            "purple",
            "red",
            "rosybrown",
            "royalblue",
            "saddlebrown",
            "salmon",
            "sandybrown",
            "seagreen",
            "seashell",
            "sienna",
            "silver",
            "skyblue",
            "slateblue",
            "snow",
            "springgreen",
            "steelblue",
            "tan",
            "teal",
            "thistle",
            "tomato",
            "turquoise",
            "violet",
            "wheat",
            "white",
            "yellow",
            "yellowgreen",
        };


    void close_stream(std::ofstream& graph)
    {
        graph.close();
    }

    std::string get_node_id(const program_node* ptr)
    {
        return "node_" + std::to_string(reinterpret_cast<uintptr_t>(ptr));
    }

    void dump_full_node(std::ofstream& out, const program_node* node)
    {
        out << node->type()->to_string(*node);
    }
    }

    std::string get_dir_path(build_options opts)
    {
        auto path = opts.get<build_option_type::graph_dumps_dir>()->directory_path;
        if (path.empty())
        {
            return{};
        }

        if (path.back() != '/' && path.back() != '\\')
        {
            path += "/";
        }
        return path;
    }

    /// Returns given name for serialization process.
    std::string get_serialization_network_name(build_options opts)
    {
        return opts.get<build_option_type::serialize_network>()->serialization_network_name;
    }

    std::string get_load_program_name(build_options opts)
    {
        return opts.get<build_option_type::load_program>()->load_program_name;
    }

    void dump_graph_init(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter)
    {
        const auto extr_oformat = [](program_node* ptr)
        {
            std::string out = fmt_to_str(ptr->get_output_layout().format);

            if (!ptr->is_valid_output_layout())
                out += " (invalid)";

            return out;
        };

        const auto dump_mem_info = [](program_node* ptr)
        {
            std::string out = "size_info: ";
            auto out_layout = ptr->get_output_layout();
            auto tensor_str = out_layout.size.to_string();
            auto padding = out_layout.data_padding;
            out += tensor_str;
            if (!padding)
            {
                out += " (nonpadded)";
            }
            else
            {
                out += "\nl: " + padding.lower_size().to_string()
                    + "\nu: " + padding.upper_size().to_string();
            }
            
            return out;
        };

        graph << "digraph cldnn_program {\n";
        for (auto& node : program.get_processing_order())
        {
            if (filter && !filter(*node))
            {
                continue;
            }
            #ifdef __clang__
                #pragma clang diagnostic push
                #pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
            #endif
            auto& node_type = typeid(*node);
            std::string node_type_name = get_extr_type(node_type.name());
            graph << "    " << get_node_id(node) << "[label=\"" << node->id() << ":\n" << node_type_name << "\n out format: " + extr_oformat(node)
                << "\n out data_type: " + dt_to_str(node->get_output_layout().data_type)
                << "\\nprocessing number: " << program.get_processing_order().get_processing_number(node) << "\\n color:" << (node->is_reusing_memory() ? std::to_string(node->get_reused_memory_color()) : "none")
                << (node->can_be_optimized() ? "\\n optimized out" : "");

            if (node_type_name != "struct cldnn::data" && node_type_name != "struct cldnn::input_layout" && !node->can_be_optimized())
            {
                graph << "\\n Selected kernel: " << (node->get_selected_impl() == nullptr ? "none" : node->get_selected_impl().get()->get_kernel_name()
                    + "\n" + dump_mem_info(node));
            }
            graph << "\"";
            #ifdef __clang__
                #pragma clang diagnostic pop
            #endif

            if (node->is_type<condition>())
            {
                graph << ", shape=diamond";
            }
            if (node->is_type<data>() || node->is_constant())
            {
                graph << ", shape=box";
            }
            if (node->is_type<internal_primitive>())
            {
                graph << ", color=blue";
            }

            if (node->is_reusing_memory())
            {
                graph << ", fillcolor=\"" << colors[node->get_reused_memory_color() % colors.size()] << "\" ";
                graph << " style=filled ";
            }
            graph << "];\n";

            for (auto& user : node->get_users())
            {
                if (filter && !filter(*user))
                {
                    continue;
                }
                bool doubled = true;
                if (std::find(user->get_dependencies().begin(), user->get_dependencies().end(), node) == user->get_dependencies().end())
                    doubled = false;
                graph << "    " << get_node_id(node) << " -> " << get_node_id(user);

                bool data_flow = node->is_in_data_flow() && user->is_in_data_flow();
                if (data_flow)
                {
                    if (doubled)
                       graph << " [color=red]";
                    else
                       graph << " [color=red, style=dashed, label=\"usr\"]";
                }
                else
                {
                    if (!doubled)
                        graph << " [style=dashed, label=\"usr\"]";
                }
                graph << ";\n";
            }

            for (auto& dep : node->get_dependencies())
            {
                if (filter && !filter(*dep))
                {
                    continue;
                }

                if (std::find(dep->get_users().begin(), dep->get_users().end(), node) != dep->get_users().end())
                {
                    continue;
                }

                graph << "   " << get_node_id(node) << " -> " << get_node_id(dep) << " [style=dashed, label=\"dep\", constraint=false];\n";
            }
        }
        graph << "}\n";
        close_stream(graph);
    }


    void dump_graph_processing_order(std::ofstream& graph, const program_impl& program)
    { 
        for (auto node : program.get_processing_order())
            graph << reinterpret_cast<uintptr_t>(node) << " (" << node->id() << ")\n";
        graph << '\n';
        close_stream(graph);
    }

    void dump_graph_optimized(std::ofstream& graph, const program_impl& program)
    {
        for (auto& prim_id : program.get_optimized_out())
            graph << prim_id << "\n";
        graph << '\n';
        close_stream(graph);
    }

    void dump_graph_info(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter)
    {
        for (auto& node : program.get_processing_order())
        {
            if (filter && !filter(*node))
                continue;

            dump_full_node(graph, node);
            graph << std::endl << std::endl;
        }
        close_stream(graph);
    }
}

 