/*
// Copyright (c) 2016 Intel Corporation
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

    std::string get_node_id(program_node* ptr)
    {
        return "node_" + std::to_string(reinterpret_cast<uintptr_t>(ptr));
    }

    void dump_full_node(std::ofstream& out, program_node* node)
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
            std::string out = "";
            switch (ptr->get_output_layout().format)
            {
            case format::yxfb: out = "yxfb"; break;
            case format::byxf: out = "byxf"; break;
            case format::bfyx: out = "bfyx"; break;
            case format::fyxb: out = "fyxb"; break;
            case format::os_iyx_osv16: out = "os_iyx_osv16"; break;
            case format::bs_xs_xsv8_bsv8: out = "bs_xs_xsv8_bsv8"; break;
            case format::bs_xs_xsv8_bsv16: out = "bs_xs_xsv8_bsv16"; break;
            case format::bs_x_bsv16: out = "bs_x_bsv16"; break;
            case format::bf8_xy16: out = "bf8_xy16"; break;
            case format::image_2d_weights_c1_b_fyx: out = "image_2d_weights_c1_b_fyx"; break;
            case format::image_2d_weights_c4_fyx_b: out = "image_2d_weights_c4_fyx_b"; break;
            case format::image_2d_weights_winograd_6x3_s1_fbxyb: out = "image_2d_weights_winograd_6x3_s1_fbxyb"; break;
            case format::image_2d_weights_winograd_6x3_s1_xfbyb: out = "image_2d_weights_winograd_6x3_s1_xfbyb"; break;
            case format::os_is_yx_isa8_osv8_isv4: out = "os_is_yx_isa8_osv8_isv4"; break;
            case format::byxf_af32: out = "byxf_af32"; break;
            case format::any: out = "any"; break;
            default:
                out = "unk format";
                break;
            }

            if (!ptr->is_valid_output_layout())
                out += " (invalid)";

            return out;
        };

        const auto extr_data_type = [](program_node* ptr)
        {
            std::string out = "";
            switch (ptr->get_output_layout().data_type)
            {
            case data_types::i8: out = "i8"; break;
            case data_types::u8: out = "u8"; break;
            case data_types::f16: out = "f16"; break;
            case data_types::f32: out = "f32"; break;
            default:
                out = "unknown data_type";
                break;
            }
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
        for (auto& node : program.get_nodes())
        {
            if (filter && !filter(*node))
            {
                continue;
            }
            #ifdef __clang__
                #pragma clang diagnostic push
                #pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
            #endif
            std::string node_type = get_extr_type(typeid(*node).name());
            graph << "    " << get_node_id(node.get()) << "[label=\"" << node->id() << ":\n" << node_type << "\n out format: " + extr_oformat(node.get())
                << "\n out data_type: " + extr_data_type(node.get())
                << "\\nprocessing number: " << node->get_processing_num() << "\\n color:" << (node->is_reusing_memory() ? std::to_string(node->get_reused_memory_color()) : "none")
                << (node->can_be_optimized() ? "\\n optimized out" : "");
            if (node_type != "struct cldnn::data" && node_type != "struct cldnn::input_layout" && !node->can_be_optimized())
                graph << "\\n Selected kernel: " << (node->get_selected_impl() == nullptr ? "none" : node->get_selected_impl().get()->get_kernel_name()
                    + "\n" + dump_mem_info(node.get()));
            graph << "\"";
            #ifdef __clang__
                #pragma clang diagnostic pop
            #endif

            if (node->is_type<data>() || node->is_constant())
                graph << ", shape=box";
            if (node->is_type<internal_primitive>())
                graph << ", color=blue";
            if (node->is_in_data_flow())
                graph << ", group=data_flow";
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
                if (std::find(user->get_dependencies().begin(), user->get_dependencies().end(), node.get()) == user->get_dependencies().end())
                    doubled = false;

                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(user);

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

                if (std::find(dep->get_users().begin(), dep->get_users().end(), node.get()) != dep->get_users().end())
                {
                    continue;
                }

                graph << "   " << get_node_id(node.get()) << " -> " << get_node_id(dep) << " [style=dashed, label=\"dep\", constraint=false];\n";
            }

            if (node->get_dominator() && (!filter || filter(*node->get_dominator())))
                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(node->get_dominator()) << " [style=dotted, label=\"dom\", constraint=false];\n";
            if (node->get_joint() && (!filter || filter(*node->get_joint())))
                graph << "    " << get_node_id(node.get()) << " -> " << get_node_id(node->get_joint()) << " [style=dotted, label=\"p-dom\", constraint=false];\n";
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
        for (auto& node : program.get_nodes())
        {
            if (filter && !filter(*node))
                continue;

            dump_full_node(graph, node.get());
            graph << std::endl << std::endl;
        }
        close_stream(graph);
    }

    //Function used by serialization. Not working yet, in progress.
    void dump_to_xml(std::ofstream& graph, const program_impl& program, std::function<bool(program_node const&)> const& filter, std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names)
    {
        xml_composite data_container, node_container, kernels;
        auto node_number = 1;
        auto kernels_number = 1;
        auto postion = 0u;
        auto offset = 0ull;
        auto size = offsets.at(0);
        for (auto& node : program.get_nodes())
        {
            if (filter && !filter(*node))
                continue;

            std::string package_name = "node_" + std::to_string(node_number);
            auto node_info = node.get()->desc_to_xml();
            auto id = node->id();
            for (auto p = postion; p < (unsigned int)data_names.size(); p++)
            {
                    if (p != 0)
                    {
                        offset = offsets.at(p - 1);
                        size = offsets.at(p) - offsets.at(p - 1);
                    }
                    if (data_names.at(p).find("kernels") != std::string::npos)
                    {
                        node_info = kernels;
                        node_info.add("id", data_names.at(p));
                        id = "kernels";
                        package_name = "kernels_" + std::to_string(kernels_number);

                        postion++;
                        kernels_number++;
                        node_number--;
                    }
                    if (data_names.at(p).find(id) != std::string::npos)
                    {
                        node_info.add("data_offset", std::to_string(offset));
                        node_info.add("data_size", std::to_string(size));
                        node_number++;
                        break;
                    }
            }
            node_container.add(package_name, node_info); 
        }
        data_container.add("data", node_container);
        data_container.dump(graph);
        close_stream(graph);
    }

    //Function used by serialization. Not working yet, in progress.
    void dump_kernels(kernels_binaries_container program_binaries, std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names, std::ofstream& file_stream)
    {
        auto offset_temp = 0ull;
        for (unsigned int i = 0; i < (unsigned int)program_binaries.size(); i++)
        {
            for (unsigned int j = 0; j < (unsigned int)program_binaries.at(i).size(); j++)
            {
                for (unsigned int k = 0; k < (unsigned int)program_binaries.at(i).at(j).size(); k++)
                {
                    char* p = (char*)&program_binaries.at(i).at(j).at(k);
                    file_stream.write(p, sizeof(char));
                    offset_temp += sizeof(char);
                }
            }
            offsets.push_back(offset_temp);
            std::string offset_name = "kernels_part_" + std::to_string(i+1);
            data_names.push_back(offset_name);
        }
    }

    //Function used by serialization. Not working yet, in progress.
    void dump_data(memory_impl& mem, std::ofstream& stream, unsigned long long& total_offset, unsigned long long type)
    {
        auto offset = 0ull;
        char * ptr = (char*)mem.lock();
        for (unsigned int x = 0; x < (unsigned int)mem.get_layout().count(); x++)
        {
            stream.write(ptr + offset, type);
            offset += type;
        }
        mem.unlock();
        total_offset += offset;
    }
}

 