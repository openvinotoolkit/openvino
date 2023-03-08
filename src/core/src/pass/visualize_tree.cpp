// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/visualize_tree.hpp"

#include <cmath>
#include <fstream>
#include <openvino/cc/pass/itt.hpp>

#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

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
    HeightMap(std::set<Node*> initials) {
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
    std::unordered_map<Node*, int64_t> m_heights;
};

static std::string label_edge(const std::shared_ptr<Node>& /* src */,
                              const std::shared_ptr<Node>& dst,
                              size_t arg_index,
                              int64_t jump_distance) {
    std::stringstream ss;
    if (getenv_bool("OV_VISUALIZE_TREE_EDGE_LABELS")) {
        size_t output = 0;
        stringstream label_edge;
        label_edge << "[label=\" " << output << " -> " << arg_index << " \"]";
        ss << label_edge.str();
    } else if (getenv_bool("OV_VISUALIZE_TREE_EDGE_JUMP_DISTANCE")) {
        if (jump_distance > 1) {
            stringstream label_edge;
            label_edge << "[label=\"jump=" << jump_distance << "\"]";
            ss << label_edge.str();
        }
    }
    return ss.str();
}

static std::string get_attribute_values(const std::map<std::string, ov::Any>& attributes,
                                        const std::string& delimiter = ", ") {
    stringstream ss;
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

bool pass::VisualizeTree::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(VisualizeTree);
    unordered_map<Node*, HeightMap> height_maps;

    for (auto& node : f->get_ops()) {
        if (node->description() == "Result") {
            height_maps[node.get()] = HeightMap({node.get()});
        } else {
            height_maps[node.get()] = HeightMap();
        }
    }

    auto nodes = topological_sort(f->get_ops());

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        auto& node = *it;
        for (auto& output : node->outputs()) {
            for (auto& input : output.get_target_inputs()) {
                auto target_node = input.get_node();
                height_maps[node.get()].absorb(height_maps[target_node]);
            }
        }
    }

    // TODO(amprocte): Maybe find a way to make this tunable.

    size_t fake_node_ctr = 0;

    traverse_nodes(f, [&](shared_ptr<Node> node) {
        add_node_arguments(node, height_maps, fake_node_ctr);
    });

    render();

    // Clean up local variable not to hold node pointers
    m_nodes_with_attributes.clear();

    return false;
}

pass::VisualizeTree::VisualizeTree(const string& file_name, node_modifiers_t nm, bool dot_only)
    : m_name{file_name},
      m_node_modifiers{nm},
      m_dot_only(dot_only) {}

void pass::VisualizeTree::add_node_arguments(shared_ptr<Node> node,
                                             unordered_map<Node*, HeightMap>& height_maps,
                                             size_t& fake_node_ctr) {
    static const int const_max_elements = getenv_int("OV_VISUALIZE_TREE_CONST_MAX_ELEMENTS", 7);

    size_t arg_index = 0;
    for (auto input_value : node->input_values()) {
        auto arg = input_value.get_node_shared_ptr();
        size_t jump_distance = height_maps[arg.get()].max_jump_to(height_maps[node.get()]);
        if (ov::is_type<ngraph::op::Constant>(arg) || ov::is_type<ngraph::op::Parameter>(arg)) {
            auto clone_name = "CLONE_" + to_string(fake_node_ctr);
            auto color = string("color=\"") + (arg->description() == "Parameter" ? "blue" : "black") + string("\"");
            std::vector<std::string> attributes{"shape=\"box\"",
                                                "style=\"dashed\"",
                                                color,
                                                string("label=\"") + get_node_name(arg) + string("\n") +
                                                    get_constant_value(arg, const_max_elements) + string("\"")};

            if (m_node_modifiers && !arg->output(0).get_rt_info().empty()) {
                m_node_modifiers(*arg, attributes);
            }
            m_ss << "    " << clone_name << "[";
            for (auto attr : attributes) {
                m_ss << " " << attr << " ";
            }
            m_ss << "]\n";

            m_ss << "    " << clone_name << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            fake_node_ctr++;
        } else if (jump_distance > max_jump_distance) {
            m_ss << add_attributes(arg);
            m_ss << add_attributes(node);
            auto recv_node_name = "RECV_" + to_string(fake_node_ctr);
            auto send_node_name = "SEND_" + to_string(fake_node_ctr);
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

string pass::VisualizeTree::add_attributes(shared_ptr<Node> node) {
    string rc;
    if (m_nodes_with_attributes.find(node) == m_nodes_with_attributes.end()) {
        m_nodes_with_attributes.insert(node);
        rc = get_attributes(node);
    }
    return rc;
}

static std::string pretty_partial_shape(const PartialShape& shape) {
    std::stringstream ss;

    if (shape.rank().is_dynamic()) {
        ss << "?";
    } else {
        bool first = true;

        ss << "[";
        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (!first) {
                ss << ",";
            }
            if (shape[i].is_dynamic()) {
                ss << shape[i];
            } else {
                ss << shape[i].get_length();
            }
            first = false;
        }
        ss << "]";
    }

    return ss.str();
}

template <typename T>
static std::string pretty_min_max_denormal_value(const vector<T>& values) {
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
static std::string pretty_value(const vector<T>& values, size_t max_elements) {
    std::stringstream ss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i < max_elements) {
            if (i != 0 && i % 8 == 0) {
                ss << std::endl;
            }
        } else {
            ss << "...";
            break;
        }

        const auto& value = values[i];
        if (i > 0)
            ss << ", ";
        ss << value;
    }

    const std::string additional_ss =
        getenv_bool("OV_VISUALIZE_TREE_MIN_MAX_DENORMAL") ? pretty_min_max_denormal_value(values) : "";
    if (!additional_ss.empty()) {
        ss << std::endl << "(" << additional_ss << ")";
    }
    return ss.str();
}

std::string pass::VisualizeTree::get_constant_value(std::shared_ptr<Node> node, size_t max_elements) {
    std::stringstream ss;
    ss << "{" << node->get_element_type().get_type_name() << "}";
    ss << pretty_partial_shape(node->get_output_partial_shape(0));

    if (!ngraph::op::is_constant(node))
        return ss.str();

    ss << "\nvalue: ";
    const auto constant = ov::as_type_ptr<ngraph::op::Constant>(node);
    switch (constant->get_output_element_type(0)) {
    case element::Type_t::undefined:
        ss << "[ undefined value ]";
        break;
    case element::Type_t::dynamic:
        ss << "[ dynamic value ]";
        break;
    case element::Type_t::u1:
        ss << "[ u1 value ]";
        break;
    case element::Type_t::u4:
        ss << "[ u4 value ]";
        break;
    case element::Type_t::i4:
        ss << "[ i4 value ]";
        break;
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
        ss << "[" << pretty_value(constant->cast_vector<double>(), max_elements) << "]";
        break;
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::i32:
    case element::Type_t::i64:
        ss << "[" << pretty_value(constant->cast_vector<int64_t>(), max_elements) << "]";
        break;
    case element::Type_t::boolean:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
        ss << "[" << pretty_value(constant->cast_vector<uint64_t>(), max_elements) << "]";
        break;
    }
    return ss.str();
}

string pass::VisualizeTree::get_attributes(shared_ptr<Node> node) {
    vector<string> attributes;
    attributes.push_back("shape=box");

    if (ngraph::op::is_output(node)) {
        attributes.push_back("color=crimson");
        attributes.push_back("penwidth=1.5");
    } else {
        attributes.push_back("color=black");
    }

    // Construct the label attribute
    {
        stringstream label;
        label << "label=\"" << get_node_name(node);

        static const bool nvtos =
            getenv_bool("NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES") || getenv_bool("OV_VISUALIZE_TREE_OUTPUT_SHAPES");
        static const bool nvtot =
            getenv_bool("NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES") || getenv_bool("OV_VISUALIZE_TREE_OUTPUT_TYPES");
        static const bool nvtio = getenv_bool("OV_VISUALIZE_TREE_IO");
        static const bool nvtrti = getenv_bool("OV_VISUALIZE_TREE_RUNTIME_INFO");

        if (nvtos || nvtot || nvtio) {
            if (nvtio) {
                for (const auto& input : node->inputs()) {
                    label << "\\nin" << to_string(input.get_index()) << ": ";
                    if (nvtot)
                        label << "{" << input.get_element_type().get_type_name() << "}";
                    if (nvtos)
                        label << pretty_partial_shape(input.get_partial_shape());
                    label << ": " << node->get_input_node_ptr(input.get_index())->get_name() << ": out"
                          << input.get_source_output().get_index();

                    if (nvtrti) {
                        label << get_attribute_values(input.get_rt_info());
                    }
                }
            }
            for (const auto& output : node->outputs()) {
                if (nvtio)
                    label << "\\nout" << to_string(output.get_index()) << ": ";
                if (nvtot)
                    label << "{" << output.get_element_type().get_type_name() << "}";
                if (nvtos)
                    label << pretty_partial_shape(output.get_partial_shape());

                if (nvtrti) {
                    label << get_attribute_values(output.get_rt_info());
                }
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

    stringstream ss;
    ss << "    " << node->get_name() << " [" << join(attributes, " ") << "]\n";

    return ss.str();
}

string pass::VisualizeTree::get_node_name(shared_ptr<Node> node) {
    static const bool nvtmn = getenv_bool("OV_VISUALIZE_TREE_MEMBERS_NAME");
    string rc = (nvtmn ? string("friendly_name: ") : "") + node->get_friendly_name();
    if (node->get_friendly_name() != node->get_name()) {
        rc += "\\n" + (nvtmn ? string("name: ") : "") + node->get_name();
    }
    rc += "\\n" + (nvtmn ? string("type_name: ") : "") + std::string(node->get_type_name());

    static const bool nvttn = getenv_bool("OV_VISUALIZE_TREE_TENSORS_NAME");
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
            rc += "\\n" + (nvtmn ? string("in_tensor_names: ") : "");
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
            rc += "\\n" + (nvtmn ? string("out_tensor_names: ") : "");
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

    static const bool nvtrti = getenv_bool("OV_VISUALIZE_TREE_RUNTIME_INFO");
    if (nvtrti) {
        const auto rt = node->get_rt_info();
        if (!rt.empty()) {
            rc += "\\nrt info: " + get_attribute_values(rt, "\\n");
        }
    }
    return rc;
}

void pass::VisualizeTree::render() const {
    NGRAPH_SUPPRESS_DEPRECATED_START
    string ext = file_util::get_file_ext(m_name);
    string output_format = ext.substr(1);
    string dot_file = m_name;
    if (to_lower(ext) != ".dot") {
        dot_file += ".dot";
    }
    ofstream out(dot_file);
    if (out) {
        out << "digraph ngraph\n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        if (!m_dot_only && to_lower(ext) != ".dot") {
#ifndef _WIN32
            stringstream ss;
            ss << "dot -T" << output_format << " " << dot_file << " -o" << m_name;
            auto cmd = ss.str();
            auto stream = popen(cmd.c_str(), "r");
            if (stream) {
                pclose(stream);
            }
#endif
        }
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
