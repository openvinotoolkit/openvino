// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_dump_graph.h"
#include "to_string_utils.h"
#include "data_inst.h"
#include "condition_inst.h"

#include <algorithm>
#include <vector>
#include <string>

namespace cldnn {
namespace {
static const std::vector<std::string> colors = {
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

void close_stream(std::ofstream& graph) { graph.close(); }

std::string get_node_id(const program_node* ptr) { return "node_" + std::to_string(reinterpret_cast<uintptr_t>(ptr)); }

void dump_full_node(std::ofstream& out, const program_node* node) { out << node->type()->to_string(*node); }
}  // namespace

std::string get_dir_path(const ExecutionConfig& config) {
    auto path = config.get_property(ov::intel_gpu::dump_graphs);
    if (path.empty()) {
        return {};
    }

    if (path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    return path;
}

void dump_graph_init(std::ofstream& graph,
                     const program& program,
                     std::function<bool(program_node const&)> const& filter) {
    const std::string invalid_layout_msg = "(invalid layout)";
    const auto extr_oformat = [&invalid_layout_msg](const program_node* ptr) {
        if (!ptr->is_valid_output_layout())
            return invalid_layout_msg;

        auto output_layout = ptr->get_output_layout();
        std::string out = output_layout.format.to_string();

        return out;
    };

    const auto extr_odt = [&invalid_layout_msg](const program_node* ptr) {
        if (!ptr->is_valid_output_layout())
            return invalid_layout_msg;

        auto output_layout = ptr->get_output_layout();
        std::string out = dt_to_str(output_layout.data_type);

        return out;
    };

    const auto dump_mem_info = [&invalid_layout_msg](const program_node* ptr) {
        std::string out = "size_info: ";
        if (!ptr->is_valid_output_layout()) {
            return out + invalid_layout_msg;
        }

        auto out_layout = ptr->get_output_layout();
        auto tensor_str = out_layout.to_string();
        auto padding = out_layout.data_padding;
        out += tensor_str;
        if (!padding) {
            out += " (nonpadded)";
        } else {
            out += "\nl: " + padding.lower_size().to_string() + "\nu: " + padding.upper_size().to_string();
        }

        return out;
    };

    graph << "digraph cldnn_program {\n";
    for (auto& node : program.get_processing_order()) {
        if (filter && !filter(*node)) {
            continue;
        }
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
#endif
        auto& node_type = typeid(*node);
        std::string node_type_name = get_extr_type(node_type.name());
        graph << "    " << get_node_id(node) << "[label=\"" << node->id() << ":\n"
              << node_type_name << "\n out format: " + extr_oformat(node)
              << "\n out data_type: " + extr_odt(node)
              << "\\nprocessing number: " << program.get_processing_order().get_processing_number(node)
              << "\\n color:" << (node->is_reusing_memory() ? std::to_string(node->get_reused_memory_color()) : "none")
              << (node->can_be_optimized() ? "\\n optimized out" : "");

        if (node_type_name != "struct cldnn::data" && node_type_name != "struct cldnn::input_layout" &&
            !node->can_be_optimized()) {
            graph << "\\n Selected kernel: "
                  << (node->get_selected_impl() == nullptr ? "none"
                                                           : node->get_selected_impl()->get_kernel_name()) + " / "
                  << node->get_preferred_impl_type()
                  << "\n" + dump_mem_info(node);
        }
        graph << "\"";
#ifdef __clang__
#pragma clang diagnostic pop
#endif

        if (node->is_type<condition>()) {
            graph << ", shape=diamond";
        }
        if (node->is_type<data>() || node->is_constant()) {
            graph << ", shape=box";
        }

        if (node->is_reusing_memory()) {
            graph << ", fillcolor=\"" << colors[node->get_reused_memory_color() % colors.size()] << "\" ";
            graph << " style=filled ";
        }
        graph << "];\n";

        for (auto& user : node->get_users()) {
            if (filter && !filter(*user)) {
                continue;
            }
            bool doubled = true;
            auto it = user->get_dependencies().begin();
            while (it != user->get_dependencies().end()) {
                if (it->first == node)
                    break;
                ++it;
            }
            if (it == user->get_dependencies().end())
                doubled = false;
            graph << "    " << get_node_id(node) << " -> " << get_node_id(user);

            bool data_flow = node->is_in_data_flow() && user->is_in_data_flow();
            if (data_flow) {
                if (doubled)
                    graph << " [color=red]";
                else
                    graph << " [color=red, style=dashed, label=\"usr\"]";
            } else {
                if (!doubled)
                    graph << " [style=dashed, label=\"usr\"]";
            }
            graph << ";\n";
        }

        for (auto& dep : node->get_dependencies()) {
            if (filter && !filter(*dep.first)) {
                continue;
            }

            if (std::find(dep.first->get_users().begin(), dep.first->get_users().end(), node) != dep.first->get_users().end()) {
                continue;
            }

            graph << "   " << get_node_id(node) << " -> " << get_node_id(dep.first)
                  << " [style=dashed, label=\"dep\", constraint=false];\n";
        }
    }
    graph << "}\n";
    close_stream(graph);
}

void dump_graph_processing_order(std::ofstream& graph, const program& program) {
    for (auto node : program.get_processing_order())
        graph << reinterpret_cast<uintptr_t>(node) << " (" << node->id() << ")\n";
    graph << '\n';
    close_stream(graph);
}

void dump_graph_optimized(std::ofstream& graph, const program& program) {
    for (auto& prim_id : program.get_optimized_out()) graph << prim_id << "\n";
    graph << '\n';
    close_stream(graph);
}

void dump_graph_info(std::ofstream& graph,
                     const program& program,
                     std::function<bool(program_node const&)> const& filter) {
    for (auto& node : program.get_processing_order()) {
        if (filter && !filter(*node))
            continue;

        dump_full_node(graph, node);
        graph << std::endl << std::endl;
    }
    close_stream(graph);
}
}  // namespace cldnn
