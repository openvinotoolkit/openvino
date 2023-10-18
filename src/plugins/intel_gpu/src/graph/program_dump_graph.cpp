// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_dump_graph.h"
#include "to_string_utils.h"
#include "data_inst.h"
#include "condition_inst.h"
#include "data_inst.h"
#include "json_object.h"

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

void dump_full_node(std::ofstream& out, const program_node* node) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    try {
        out << node->type()->to_string(*node);
    } catch(const std::exception& e) {
        auto node_info = std::shared_ptr<json_composite>(new json_composite());
        node_info->add("id", node->id());
        node_info->add("ptr", "node_" + std::to_string(reinterpret_cast<uintptr_t>(node)));
        node_info->add("error", "failed to make string from descriptor");
        std::stringstream emtpy_desc;
        node_info->dump(emtpy_desc);
        out << emtpy_desc.str();

        GPU_DEBUG_INFO << node->id() << " to_string() error: " << e.what() << '\n';
    }
}
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

    const auto dump_mem_info = [&invalid_layout_msg](const program_node* ptr) {
        std::string out = "layout_info: ";
        if (!ptr->is_valid_output_layout()) {
            return out + invalid_layout_msg;
        }

        auto out_layout = ptr->get_output_layout();
        if (!out_layout.data_padding) {
            out += " " +  out_layout.to_short_string();
        } else {
            out += " " + out_layout.to_string();
        }

        return out;
    };
    const auto dump_mem_preferred_info = [](const program_node* ptr) {
        std::string out = "";
        auto input_fmts = ptr->get_preferred_input_fmts();
        if (!input_fmts.empty()) {
            out += "preferred_in_fmt";
            for (auto& fmt : input_fmts) {
                out += ":" + fmt_to_str(fmt);
            }
        }
        auto output_fmts = ptr->get_preferred_output_fmts();
        if (!output_fmts.empty()) {
            out += "\npreferred_out_fmt";
            for (auto& fmt : output_fmts) {
                out += ":" + fmt_to_str(fmt);
            }
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
        std::string node_type_name = node->get_primitive()->type_string();
        graph << "    " << get_node_id(node) << "[label=\"" << node->id() << ":"
              << "\\ntype: " << node_type_name
              << "\\nprocessing number: " << program.get_processing_order().get_processing_number(node)
              << "\\n color:" << (node->is_reusing_memory() ? std::to_string(node->get_reused_memory_color()) : "none")
              << (node->can_be_optimized() ? "\\n optimized out" : "");

        if (!node->is_type<data>()) {
            graph << "\\n Selected kernel: "
                  << (node->get_selected_impl() == nullptr ? "none"
                        : (node->get_preferred_impl_type() == impl_types::ocl && node->get_selected_impl()->get_kernels_dump_info().second.size())
                        ?  node->get_selected_impl()->get_kernels_dump_info().second
                        : node->get_selected_impl()->get_kernel_name()) + " / "
                  << node->get_preferred_impl_type();
            if (node->get_selected_impl()) {
                auto dump_info = node->get_selected_impl()->get_kernels_dump_info();
                if (dump_info.first.size()) {
                    graph << "\\n batch_hash : " << dump_info.first;
                }
            }
        }
        graph << "\n" + dump_mem_info(node);
        graph << "\n" + dump_mem_preferred_info(node);
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
            graph << "    " << get_node_id(node) << " -> " << get_node_id(user)
                  << " [label=\"" << it->second << " -> " << std::distance(user->get_dependencies().begin(), it) << "\"]";


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
