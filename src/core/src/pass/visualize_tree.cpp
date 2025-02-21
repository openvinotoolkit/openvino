// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/visualize_tree.hpp"

#include <cmath>
#include <fstream>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

/*
 * As we are visualizing the graph, we will make some tweaks to the generated dot file to make
 * routing more tractable for Graphviz as well as (hopefully) more legible for the user.
 *
 * NOTE: It's possible, even likely, that better algorithms are available here. I just tried a
 * few different things without doing much research, and this seemed to work well. Please feel
 * free to improve on this. --amprocte
 *
 * -----------------
 *
 * The first tweak is to trim edges that, intuitively speaking, have long "skip distance". For
 * example:
 *
 * [Actual Graph Structure]      [Visualization]
 *    n0                             n0
 *    | \                            |  \
 *    n1 \                           n1  [to n50]
 *    |   |                          |
 *    n2  |                          n2
 *    |   |                          |
 *    n3  |                          n3
 *    |   |                          |
 *   ...  |                         ...  [from n0]
 *    |  /                           |  /
 *   n50                            n50
 *
 * This is useful for training graphs especially, which tend to have very long feed-forward edges
 * for intermediate values from fprop being stored for later reuse in the bprop phase.
 *
 * Efficiently detecting a "long skip" is a bit tricky. We want to come up with a metric that is
 * reasonably fast to compute, but does not result in cuts that will split the graph into multiple
 * components. The heuristic we are using for the jump distance between n and m is the maximum
 * difference in maximum path length from n and m to any result node that is reachable from both
 * n and m (or 0, if no such result node exists). Not sure if this is mathematically *guaranteed*
 * not to split graph components, but it seems to work well in practice.
 *
 * Formally:
 *
 * Compute-Heights-Above-Each-Parameter(N):
 *    Inputs: nodes N; define R={n in N | n is a Result node}
 *    Output: height_maps: map from N to (map from R to int)
 *
 *    height_maps is initially empty
 *
 *    for each r in R:
 *        Insert into height_map the map {r -> 1}
 *
 *    for each n in N in reverse topological ("results-first") order:
 *        for each user m of n:
 *            for each r in height_maps[m].keys:
 *                height_maps[n][r] := max(height_maps[n][r], height_maps[m][r]+1)
 *
 * Jump-Distance(n,m,height_maps):
 *     Inputs: n (source node), m (destination node), height_maps (pre-computed above)
 *     Output: jump_distance: int
 *
 *     jump_distance := 0
 *
 *     for each r in height_maps[n].keys:
 *         if r is in height_maps[m].keys:
 *             jump_distance := max(jump_distance, abs(height_maps[n][r] - height_maps[m][r]))
 *
 * Later on, if E is an edge from n to m, and Jump-Distance(n,m,height_map) > K (where K is kind
 * of arbitrary but currently set to 20), we will "cut" the edge as illustrated above.
 *
 * -----------------
 *
 * The second tweak aims to eliminate routing pressure from nodes that have large outdegree and
 * are connected to many otherwise-distant places in the graph. For this, the only thing we are
 * doing at the moment is to "float" Parameter and Constant nodes. This means that rather than
 * visualizing them as a single node (which might have very large outdegree as in, e.g., a
 * learning rate parameter being fed to many different places), we make a "copy" of the node at
 * each occurrence site (with a dashed outline).
 *
 * NOTE: This tweak could probably be extended to float other kinds of nodes with high out-degree.
 * (This situation is likely to arise after constant subexpression elimination.) Here one has to
 * be careful to avoid splitting the components. I have some rough ideas on how this could be
 * dealt with, but have not had time to implement them yet. --amprocte
 */

class HeightMap {
public:
    HeightMap() {}
    HeightMap(std::set<ov::Node*> initials) {
        for (auto& n : initials) {
            m_heights[n] = 0;
        }
    }
    void absorb(const HeightMap& other) {
        for (auto& p : other.m_heights) {
            auto k = p.first;
            auto v = p.second;
            m_heights[k] = std::max(m_heights[k], v + 1);
        }
    }
    int64_t max_jump_to(const HeightMap& target) {
        int64_t result = 0;
        for (auto& p : m_heights) {
            auto k = p.first;
            auto v = p.second;
            if (target.m_heights.count(k) != 0) {
                result = std::max(result, std::abs(target.m_heights.at(k) - v));
            }
        }
        return result;
    }

private:
    std::unordered_map<ov::Node*, int64_t> m_heights;
};

static std::string label_edge(const std::shared_ptr<ov::Node>& /* src */,
                              const std::shared_ptr<ov::Node>& dst,
                              size_t arg_index,
                              int64_t jump_distance) {
    std::stringstream ss;
    if (ov::util::getenv_bool("OV_VISUALIZE_TREE_EDGE_LABELS")) {
        ss << "[label=\" " << dst->input_value(arg_index).get_index() << " -> " << arg_index << " \"]";
    } else if (ov::util::getenv_bool("OV_VISUALIZE_TREE_EDGE_JUMP_DISTANCE")) {
        if (jump_distance > 1) {
            ss << "[label=\"jump=" << jump_distance << "\"]";
        }
    }
    return ss.str();
}

static std::string get_attribute_values(const std::map<std::string, ov::Any>& attributes,
                                        const std::string& delimiter = ", ") {
    std::stringstream ss;
    bool first = true;
    for (const auto& item : attributes) {
        ss << (first ? " " : delimiter) << item.first;
        if (item.second.is<ov::RuntimeAttribute>()) {
            ss << "{" << item.second.as<ov::RuntimeAttribute>().to_string() << "}";
        } else if (!item.second.empty()) {
            ss << "{";
            item.second.print(ss);
            ss << "}";
        } else {
            ss << "{"
               << "[EMPTY]"
               << "}";
        }

        first = false;
    }
    return ss.str();
}

static std::string name_of_subgraph_file(const std::shared_ptr<ov::Node> op,
                                         const std::string& current_file_name,
                                         const size_t& i) {
    // friendly is never empty it is either friendly (set by user) or unique (auto-generated) name
    auto node_name = op->get_friendly_name();
    std::replace(node_name.begin(), node_name.end(), '/', '-');
    auto postfix = "_node_" + node_name + "_subgraph_#" + std::to_string(i);
    auto file_name = current_file_name;
    auto insert_pos = file_name.find_last_of('.');
    file_name.insert(insert_pos, postfix);
    return file_name;
}

static void collect_symbol_print_values(const std::shared_ptr<ov::Model>& m,
                                        std::unordered_map<std::shared_ptr<ov::Symbol>, size_t>& symbol_to_number) {
    size_t n = symbol_to_number.size() + 1;
    for (const auto& node : m->get_ops()) {
        if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node))
            for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i)
                if (const auto& sub_graph = multi_subgraph_op->get_function(i))
                    collect_symbol_print_values(sub_graph, symbol_to_number);

        for (const auto& output : node->outputs()) {
            const auto& shape = output.get_partial_shape();
            if (shape.rank().is_dynamic())
                continue;
            for (const auto& dim : shape)
                if (auto symbol = dim.get_symbol()) {
                    const auto& root = ov::symbol::ancestor_of(symbol);
                    if (symbol_to_number.count(root))
                        continue;
                    symbol_to_number[root] = n++;
                }
            const auto& value_symbols = output.get_tensor().get_value_symbol();
            for (const auto& value_symbol : value_symbols)
                if (value_symbol) {
                    const auto& root = ov::symbol::ancestor_of(value_symbol);
                    if (symbol_to_number.count(root))
                        continue;
                    symbol_to_number[root] = n++;
                }
        }
    }
}

bool ov::pass::VisualizeTree::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(VisualizeTree);

    static const bool ovasp = ov::util::getenv_bool("OV_VISUALIZE_APPLY_SYMBOLIC_PROPAGATION");
    if (ovasp) {
        std::cerr << "Warning: OV_VISUALIZE_APPLY_SYMBOLIC_PROPAGATION enabled. ov::pass::SymbolicPropagation will be "
                     "triggered"
                  << std::endl;
        ov::pass::SymbolicPropagation().run_on_model(f);
        std::cerr << "ov::pass::SymbolicPropagation finished successfully" << std::endl;
    }

    std::unordered_map<Node*, HeightMap> height_maps;

    for (auto& node : f->get_ops()) {
        if (node->description() == "Result") {
            height_maps[node.get()] = HeightMap({node.get()});
        } else {
            height_maps[node.get()] = HeightMap();
        }
    }

    auto nodes = topological_sort(f->get_ops());

    collect_symbol_print_values(f, m_symbol_to_name);

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        auto& node = *it;
        if (auto multi_subgraph_op = ov::as_type_ptr<op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i)
                if (const auto& sub_graph = multi_subgraph_op->get_function(i))
                    ov::pass::VisualizeTree(name_of_subgraph_file(multi_subgraph_op, m_name, i),
                                            m_node_modifiers,
                                            m_dot_only)
                        .run_on_model(sub_graph);
        }
        for (auto& output : node->outputs()) {
            for (auto& input : output.get_target_inputs()) {
                auto target_node = input.get_node();
                height_maps[node.get()].absorb(height_maps[target_node]);
            }
        }
    }

    // TODO(amprocte): Maybe find a way to make this tunable.

    size_t fake_node_ctr = 0;

    traverse_nodes(f, [&](const std::shared_ptr<Node>& node) {
        add_node_arguments(node, height_maps, fake_node_ctr);
    });

    render();

    // Clean up local variable not to hold node pointers
    m_nodes_with_attributes.clear();
    if (ovasp) {
        std::cerr << "Warning: Due to previously triggered SymbolicPropagation we need to clean-up the model from "
                     "symbols. It includes model revalidation"
                  << std::endl;
        ov::remove_skip_invalidation_rti(f);
        std::cerr << "Model revalidation finished successfully" << std::endl;
    }
    return false;
}

ov::pass::VisualizeTree::VisualizeTree(const std::string& file_name, node_modifiers_t nm, bool dot_only)
    : m_name{file_name},
      m_node_modifiers{std::move(nm)},
      m_dot_only{dot_only} {}

void ov::pass::VisualizeTree::add_node_arguments(std::shared_ptr<Node> node,
                                                 std::unordered_map<Node*, HeightMap>& height_maps,
                                                 size_t& fake_node_ctr) {
    size_t arg_index = 0;
    for (const auto& input_value : node->input_values()) {
        auto arg = input_value.get_node_shared_ptr();
        size_t jump_distance = height_maps[arg.get()].max_jump_to(height_maps[node.get()]);
        if (ov::is_type<ov::op::v0::Constant>(arg) || ov::is_type<ov::op::v0::Parameter>(arg)) {
            auto clone_name = "CLONE_" + std::to_string(fake_node_ctr);
            auto color =
                std::string("color=\"") + (arg->description() == "Parameter" ? "blue" : "black") + std::string("\"");
            std::vector<std::string> attributes{"shape=\"box\"",
                                                "style=\"dashed\"",
                                                std::move(color),
                                                std::string("label=\"") + get_node_name(arg) + std::string("\n") +
                                                    get_constant_value(arg) + std::string("\"")};

            if (m_node_modifiers && !arg->output(0).get_rt_info().empty()) {
                m_node_modifiers(*arg, attributes);
            }
            m_ss << "    " << clone_name << "[";
            for (const auto& attr : attributes) {
                m_ss << " " << attr << " ";
            }
            m_ss << "]\n";

            m_ss << "    " << clone_name << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            fake_node_ctr++;
        } else if (jump_distance > max_jump_distance) {
            m_ss << add_attributes(arg);
            m_ss << add_attributes(node);
            auto recv_node_name = "RECV_" + std::to_string(fake_node_ctr);
            auto send_node_name = "SEND_" + std::to_string(fake_node_ctr);
            m_ss << "    " << recv_node_name
                 << "[shape=\"box\" style=\"solid,filled\" "
                    "fillcolor=\"#ffcccc\" label=\"Receive["
                 << arg->get_name() << "]\"]\n";
            m_ss << "    " << send_node_name
                 << "[shape=\"box\" style=\"solid,filled\" "
                    "fillcolor=\"#ccffcc\" label=\"Send["
                 << node->get_name() << "]\"]\n";
            m_ss << "    " << arg->get_name() << " -> " << send_node_name
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            m_ss << "    " << recv_node_name << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            fake_node_ctr++;
        } else {
            m_ss << add_attributes(arg);
            m_ss << add_attributes(node);
            m_ss << "    " << arg->get_name() << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
        }
        arg_index++;
    }
}

std::string ov::pass::VisualizeTree::add_attributes(std::shared_ptr<Node> node) {
    std::string rc;
    if (m_nodes_with_attributes.find(node) == m_nodes_with_attributes.end()) {
        rc = get_attributes(node);
        m_nodes_with_attributes.insert(std::move(node));
    }
    return rc;
}

static std::string pretty_partial_shape(
    const ov::PartialShape& shape,
    const std::unordered_map<std::shared_ptr<ov::Symbol>, size_t>& symbol_map = {}) {
    std::stringstream str;
    if (shape.rank().is_static()) {
        str << "[";
        bool first = true;
        for (auto& d : shape) {
            if (!first) {
                str << ",";
            }
            if (d.is_dynamic()) {
                if (const auto& symbol = d.get_symbol()) {
                    const auto& root = ov::symbol::ancestor_of(symbol);
                    if (symbol_map.count(root))
                        str << "<" << symbol_map.at(root) << ">";
                    else
                        str << "<?>";
                }
            }
            str << d;
            first = false;
        }
        str << "]";
    } else {
        str << "[...]";
    }
    return str.str();
}

template <typename T>
static std::string pretty_min_max_denormal_value(const std::vector<T>& values) {
    std::stringstream ss;

    T min_value = values[0];
    T max_value = values[0];
    size_t denormal_counts = 0ul;
    std::stringstream denormal_ss;
    for (size_t i = 0; i < values.size(); ++i) {
        const auto& value = values[i];
        if (min_value > value) {
            min_value = value;
        }
        if (max_value < value) {
            max_value = value;
        }

        const auto abs_value = std::abs(static_cast<double>(value));
        if (((abs_value > 0.) && (abs_value < 1.e-32)) || (abs_value > 1.e+32)) {
            if (denormal_counts < 3) {
                denormal_ss << (denormal_counts > 0 ? ", " : "") << i << ": " << value;
            } else if (denormal_counts == 3) {
                denormal_ss << "...";
            }
            denormal_counts++;
        }
    }

    ss << "min: " << min_value << ", max: " << max_value;
    if (denormal_counts != 0) {
        ss << ", denormals: " << denormal_counts << " [" << denormal_ss.str() << "]";
    }

    return ss.str();
}

template <typename T>
static std::string pretty_value(const std::vector<T>& values, bool allow_obfuscate = false) {
    std::stringstream ss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0 && i % 8 == 0)
            ss << std::endl;
        const auto& value = values[i];
        if (i > 0)
            ss << ", ";
        if (allow_obfuscate && value == std::numeric_limits<T>::max())
            ss << "max";
        else if (allow_obfuscate && value == std::numeric_limits<T>::min())
            ss << "min";
        else
            ss << value;
    }

    const std::string additional_ss =
        ov::util::getenv_bool("OV_VISUALIZE_TREE_MIN_MAX_DENORMAL") ? pretty_min_max_denormal_value(values) : "";
    if (!additional_ss.empty()) {
        ss << std::endl << "(" << additional_ss << ")";
    }
    return ss.str();
}

static std::string get_value(const std::shared_ptr<ov::op::v0::Constant>& constant, bool allow_obfuscate = false) {
    static const int max_elements = ov::util::getenv_int("OV_VISUALIZE_TREE_CONST_MAX_ELEMENTS", 7);
    std::stringstream ss;
    ss << "[ ";
    switch (constant->get_output_element_type(0)) {
    case ov::element::Type_t::dynamic:
    case ov::element::Type_t::u1:
    case ov::element::Type_t::u2:
    case ov::element::Type_t::u3:
    case ov::element::Type_t::u4:
    case ov::element::Type_t::u6:
    case ov::element::Type_t::nf4:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::f8e4m3:
    case ov::element::Type_t::f8e5m2:
    case ov::element::Type_t::f4e2m1:
    case ov::element::Type_t::f8e8m0:
        ss << constant->get_output_element_type(0).get_type_name() << " value";
        break;
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f64:
        ss << pretty_value(constant->cast_vector<double>(max_elements), allow_obfuscate);
        break;
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
        ss << pretty_value(constant->cast_vector<int64_t>(max_elements), allow_obfuscate);
        break;
    case ov::element::Type_t::boolean:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
        ss << pretty_value(constant->cast_vector<uint64_t>(max_elements), allow_obfuscate);
        break;
    case ov::element::Type_t::string:
        ss << constant->get_output_element_type(0).get_type_name() << " value";
        break;
    }
    const auto num_elements_in_constant = static_cast<int>(shape_size(constant->get_shape()));
    if (num_elements_in_constant == 0)
        ss << "empty";
    else if (max_elements == 0)
        ss << "suppressed";
    else if (num_elements_in_constant > max_elements)
        ss << ", ...";
    ss << " ]";
    return ss.str();
}

static std::string pretty_symbol_value(const ov::TensorSymbol& symbols,
                                       const std::unordered_map<std::shared_ptr<ov::Symbol>, size_t>& symbol_map = {}) {
    std::vector<size_t> mapped_symbols;
    for (const auto& symbol : symbols) {
        if (symbol) {
            const auto& root = ov::symbol::ancestor_of(symbol);
            if (symbol_map.count(root)) {
                mapped_symbols.push_back(symbol_map.at(root));
                continue;
            }
        }
        mapped_symbols.push_back(0);
    }
    return pretty_value(mapped_symbols);
}

static std::string get_bounds_and_label_info(
    const ov::Output<ov::Node> output,
    const std::unordered_map<std::shared_ptr<ov::Symbol>, size_t>& symbol_map = {}) {
    const auto& tensor = output.get_tensor();
    const auto& lower = tensor.get_lower_value();
    const auto& upper = tensor.get_upper_value();
    const auto& value_symbol = tensor.get_value_symbol();

    if (!lower && !upper && value_symbol.empty())
        return "";

    std::stringstream label;
    size_t size = lower ? lower.get_size() : upper ? upper.get_size() : value_symbol.size();
    if (size == 0) {
        label << "empty";
    } else {
        label << " lower: " << (lower ? get_value(std::make_shared<ov::op::v0::Constant>(lower), true) : "NONE");
        label << " upper: " << (upper ? get_value(std::make_shared<ov::op::v0::Constant>(upper), true) : "NONE");
        label << " symbl: " << (value_symbol.empty() ? "NONE" : pretty_symbol_value(value_symbol, symbol_map));
    }
    return label.str();
}

std::string ov::pass::VisualizeTree::get_constant_value(std::shared_ptr<Node> node, size_t max_elements) {
    std::stringstream ss;
    ss << "{" << node->get_element_type().to_string() << "}";
    ss << pretty_partial_shape(node->get_output_partial_shape(0), m_symbol_to_name);

    if (const auto& constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        ss << "\nvalue: " << get_value(constant, max_elements);
    }
    return ss.str();
}

std::string ov::pass::VisualizeTree::get_attributes(std::shared_ptr<Node> node) {
    std::vector<std::string> attributes;
    attributes.push_back("shape=box");

    if (ov::op::util::is_output(node)) {
        attributes.push_back("color=crimson");
        attributes.push_back("penwidth=1.5");
    } else {
        attributes.push_back("color=black");
    }

    // Construct the label attribute
    {
        std::stringstream label;
        label << "label=\"" << get_node_name(node);

        static const bool nvtos = ov::util::getenv_bool("OV_VISUALIZE_TREE_OUTPUT_SHAPES");
        static const bool nvtot = ov::util::getenv_bool("OV_VISUALIZE_TREE_OUTPUT_TYPES");
        static const bool nvtio = ov::util::getenv_bool("OV_VISUALIZE_TREE_IO");
        static const bool nvtrti = ov::util::getenv_bool("OV_VISUALIZE_TREE_RUNTIME_INFO");
        static const bool ovpvl = ov::util::getenv_bool("OV_VISUALIZE_PARTIAL_VALUES_AND_LABELS");

        if (nvtos || nvtot || nvtio) {
            if (nvtio) {
                for (const auto& input : node->inputs()) {
                    label << "\\nin" << std::to_string(input.get_index()) << ": ";
                    if (nvtot)
                        label << "{" << input.get_element_type().to_string() << "}";
                    if (nvtos)
                        label << pretty_partial_shape(input.get_partial_shape(), m_symbol_to_name);
                    label << ": " << node->get_input_node_ptr(input.get_index())->get_name() << ": out"
                          << input.get_source_output().get_index();

                    if (nvtrti) {
                        label << get_attribute_values(input.get_rt_info());
                    }
                }
            }
            for (const auto& output : node->outputs()) {
                if (nvtio)
                    label << "\\nout" << std::to_string(output.get_index()) << ": ";
                if (nvtot)
                    label << "{" << output.get_element_type().to_string() << "}";
                if (nvtos)
                    label << pretty_partial_shape(output.get_partial_shape(), m_symbol_to_name);  // TODO

                if (nvtrti) {
                    label << get_attribute_values(output.get_rt_info());
                }
                if (ovpvl)
                    label << get_bounds_and_label_info(output, m_symbol_to_name);  // TODO
            }
        }

        auto eh = m_ops_to_details.find(node->get_type_info());
        if (eh != m_ops_to_details.end()) {
            eh->second(*node, label);
        }
        label << "\"";
        attributes.push_back(label.str());
    }

    if (m_node_modifiers) {
        m_node_modifiers(*node, attributes);
    }

    std::stringstream ss;
    ss << "    " << node->get_name() << " [" << ov::util::join(attributes, " ") << "]\n";

    return ss.str();
}

std::string ov::pass::VisualizeTree::get_node_name(std::shared_ptr<Node> node) {
    static const bool nvtmn = ov::util::getenv_bool("OV_VISUALIZE_TREE_MEMBERS_NAME");
    std::string rc = (nvtmn ? std::string("friendly_name: ") : "") + node->get_friendly_name();
    if (node->get_friendly_name() != node->get_name()) {
        rc += "\\n" + (nvtmn ? std::string("name: ") : "") + node->get_name();
    }
    const auto& type_info = node->get_type_info();
    rc += "\\n" + (nvtmn ? std::string("type_name: ") : "") + std::string(type_info.version_id) +
          "::" + std::string(type_info.name);

    static const bool nvttn = ov::util::getenv_bool("OV_VISUALIZE_TREE_TENSORS_NAME");
    if (nvttn) {
        auto to_string = [](const std::unordered_set<std::string>& names) {
            std::stringstream ss;
            size_t i = 0;
            for (const auto& name : names) {
                ss << (i == 0 ? "" : ", ") << name;
                i++;
            }
            return ss.str();
        };

        if (node->get_input_size() != 0) {
            rc += "\\n" + (nvtmn ? std::string("in_tensor_names: ") : "");
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                const auto input = node->input(i);
                const auto tensor_ptr = input.get_tensor_ptr();
                rc += (i == 0 ? "" : "; ") + std::string("(") + std::to_string((size_t)tensor_ptr.get()) + ") ";
                const auto str = to_string(node->input_value(0).get_names());
                if (!str.empty()) {
                    rc += str;
                }
            }
        }
        if (node->get_output_size() != 0) {
            rc += "\\n" + (nvtmn ? std::string("out_tensor_names: ") : "");
            for (size_t i = 0; i < node->get_output_size(); ++i) {
                const auto output = node->output(i);
                const auto tensor_ptr = output.get_tensor_ptr();
                rc += (i == 0 ? "" : "; ") + std::string("(") + std::to_string((size_t)tensor_ptr.get()) + ") ";
                const auto str = to_string(output.get_names());
                if (!str.empty()) {
                    rc += str;
                }
            }
        }
    }

    static const bool nvtrti = ov::util::getenv_bool("OV_VISUALIZE_TREE_RUNTIME_INFO");
    if (nvtrti) {
        const auto& rt = node->get_rt_info();
        if (!rt.empty()) {
            rc += "\\nrt info: " + get_attribute_values(rt, "\\n");
        }
    }
    return rc;
}

void ov::pass::VisualizeTree::render() const {
    std::string ext = ov::util::get_file_ext(m_name);
    std::string output_format = ext.substr(1);
    std::string dot_file = m_name;
    if (ov::util::to_lower(ext) != ".dot") {
        dot_file += ".dot";
    }
    std::ofstream out(dot_file);
    if (out) {
        out << "digraph \n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        if (!m_dot_only && ov::util::to_lower(ext) != ".dot") {
#if defined(ENABLE_OPENVINO_DEBUG) && !defined(_WIN32)
            std::stringstream ss;
            if (system("command -v dot > /dev/null 2>&1") != 0) {
                OPENVINO_THROW("Graphviz 'dot' command not found in PATH");
            }
            ss << "dot -T" << output_format << " " << dot_file << " -o" << m_name;
            auto cmd = ss.str();
            auto stream = popen(cmd.c_str(), "r");
            if (stream) {
                pclose(stream);
            }
#endif
        }
    }
}
