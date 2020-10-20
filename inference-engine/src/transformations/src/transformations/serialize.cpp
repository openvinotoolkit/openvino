// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <fstream>
#include <map>
#include <unordered_set>

#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset.hpp"
#include "pugixml.hpp"
#include "transformations/serialize.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::Serialize, "Serialize", 0);

namespace {  // helpers
template <typename T, typename A>
std::string joinVec(std::vector<T, A> const& vec,
                    std::string const& glue = std::string(",")) {
    if (vec.empty()) return "";
    std::stringstream oss;
    oss << vec[0];
    for (size_t i = 1; i < vec.size(); i++) oss << glue << vec[i];
    return oss.str();
}
}  // namespace

namespace {  // implementation details
struct Edge {
    int from_layer = 0;
    int from_port = 0;
    int to_layer = 0;
    int to_port = 0;

    Edge() = default;
    Edge(int a, int b, int c, int d)
        : from_layer{a}, from_port{b}, to_layer{c}, to_port{d} {}
};

struct ConstantAtributes {
    int size = 0;
    int offset = 0;
};

template <typename T>
std::string create_atribute_list(
    ngraph::ValueAccessor<std::vector<T>>& adapter) {
    return joinVec(adapter.get(), std::string(","));
}

class XmlVisitor : public ngraph::AttributeVisitor {
    pugi::xml_node m_data;

public:
    XmlVisitor(pugi::xml_node& data) : m_data(data) {}

    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<void>& adapter) override {
#if 0  // TODO: remove when Constant will support VisitorAPI
        m_data.append_attribute(name.c_str());
#endif
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<bool>& adapter) override {
        m_data.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<std::string>& adapter) override {
        m_data.append_attribute(name.c_str()).set_value(adapter.get().c_str());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<int64_t>& adapter) override {
        m_data.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<double>& adapter) override {
        m_data.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_data.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_data.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_data.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_data.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
};

std::map<ngraph::Node*, int> create_layer_ids(const ngraph::Function& f) {
    std::map<ngraph::Node*, int> layer_ids;
    int id = 0;
    for (auto node : f.get_ordered_ops()) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

std::vector<Edge> create_edge_mapping(std::map<ngraph::Node*, int>& layer_ids,
                                      const ngraph::Function& f) {
    std::vector<Edge> edges;
    for (auto node : f.get_ordered_ops()) {
        if (dynamic_cast<ngraph::op::Parameter*>(node.get()) != nullptr) {
            continue;
        }

        for (auto i : node->inputs()) {
            auto source_output = i.get_source_output();
            auto source_node = source_output.get_node();
            auto current_node = i.get_node();

            Edge e{};
            e.from_layer = layer_ids[source_node];
            e.from_port =
                source_node->get_input_size() + source_output.get_index();
            e.to_layer = layer_ids[current_node];
            e.to_port = i.get_index();
            edges.push_back(e);
        }
    }
    std::sort(begin(edges), end(edges),
              [](const Edge& a, const Edge& b) -> bool {
                  return a.from_layer < b.from_layer;
              });
    return edges;
}

ConstantAtributes dump_constant_data(std::vector<uint8_t>& bin,
                                     const ngraph::op::Constant& c) {
    ConstantAtributes attr;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(c.get_data_ptr());
    attr.size = ngraph::shape_size(c.get_shape()) * c.get_element_type().size();
    attr.offset = bin.size();
    bin.insert(end(bin), p, p + attr.size);
    return attr;
}

std::string get_opset_name(const ngraph::Node* n) {
    auto opsets = std::array<std::reference_wrapper<const ngraph::OpSet>, 5>{
        ngraph::get_opset1(), ngraph::get_opset2(), ngraph::get_opset3(),
        ngraph::get_opset4(), ngraph::get_opset5()};

    // return the oldest opset name where node type is present
    for (int idx = 0; idx < opsets.size(); idx++) {
        int number = idx + 1;
        if (opsets[idx].get().contains_op_type(n)) {
            return "opset" + std::to_string(number);
        }
    }
    return "unknown";
}

std::string get_type_name(const ngraph::Node* n) {
    const std::map<std::string, std::string> translator = {
        {"Constant", "Const"}};
    std::string name = n->get_type_name();
    if (translator.count(name) > 0) {
        name = translator.at(name);
    }
    return name;
}

std::string get_output_precision_name(ngraph::Output<Node>& o) {
    auto elem_type = o.get_element_type();
    switch (elem_type) {
    case ::ngraph::element::Type_t::undefined:
        return "UNSPECIFIED";
    case ::ngraph::element::Type_t::f16:
        return "FP16";
    case ::ngraph::element::Type_t::f32:
        return "FP32";
    case ::ngraph::element::Type_t::bf16:
        return "BF16";
    case ::ngraph::element::Type_t::i8:
        return "I8";
    case ::ngraph::element::Type_t::i16:
        return "I16";
    case ::ngraph::element::Type_t::i32:
        return "I32";
    case ::ngraph::element::Type_t::i64:
        return "I64";
    case ::ngraph::element::Type_t::u8:
        return "U8";
    case ::ngraph::element::Type_t::u16:
        return "U16";
    case ::ngraph::element::Type_t::u32:
        return "U32";
    case ::ngraph::element::Type_t::u64:
        return "U64";
    case ::ngraph::element::Type_t::u1:
        return "BIN";
    case ::ngraph::element::Type_t::boolean:
        return "BOOL";
    default:
        NGRAPH_CHECK(false, "Incorrect precision: ", elem_type.get_type_name());
        return "";
    }
}

std::string generate_unique_name(
    const std::unordered_set<std::string>& unique_names, std::string name,
    int suffix) {
    std::string new_name = name + std::to_string(suffix);
    if (unique_names.find(new_name) == unique_names.end()) {
        return new_name;
    } else {
        suffix++;
        return generate_unique_name(unique_names, name, suffix);
    }
}

// TODO: remove when CNNNetwork will be suporting not-unique names
std::string get_node_unique_name(std::unordered_set<std::string>& unique_names,
                                 const ngraph::Node* n) {
    std::string name = n->get_friendly_name();
    if (unique_names.find(name) != unique_names.end()) {
        name = generate_unique_name(unique_names, name, 0);
    }
    unique_names.insert(name);
    return name;
}

void ngfunction_2_irv10(pugi::xml_document& doc, std::vector<uint8_t>& bin,
                        const ngraph::Function& f) {
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(f.get_friendly_name().c_str());
    netXml.append_attribute("version").set_value("10");
    pugi::xml_node layers = netXml.append_child("layers");

    auto layer_ids = create_layer_ids(f);
    std::unordered_set<std::string> unique_names;

    for (auto node : f.get_ordered_ops()) {
        // <layers>
        pugi::xml_node layer = layers.append_child("layer");
        layer.append_attribute("id").set_value(layer_ids[node.get()]);
        layer.append_attribute("name").set_value(
            get_node_unique_name(unique_names, node.get()).c_str());
        layer.append_attribute("type").set_value(
            get_type_name(node.get()).c_str());
        layer.append_attribute("version").set_value(
            get_opset_name(node.get()).c_str());

        // <layers/data>
        pugi::xml_node data = layer.append_child("data");

        // <layers/data> general atributes
        XmlVisitor visitor{data};

        if (!node->visit_attributes(visitor)) {
            NGRAPH_CHECK(false, "Cannot visit  ", node->get_name());
        }

        // <layers/data> constant atributes (special case)
        if (auto constant = dynamic_cast<ngraph::op::Constant*>(node.get())) {
            ConstantAtributes attr = dump_constant_data(bin, *constant);
            data.append_attribute("offset").set_value(attr.offset);
            data.append_attribute("size").set_value(attr.size);
        }

        int port_id = 0;
        // <layers/input>
        if (node->get_input_size() > 0) {
            pugi::xml_node input = layer.append_child("input");
            for (auto i : node->inputs()) {
                pugi::xml_node port = input.append_child("port");
                port.append_attribute("id").set_value(port_id++);
                for (auto d : i.get_shape()) {
                    pugi::xml_node dim = port.append_child("dim");
                    dim.append_child(pugi::xml_node_type::node_pcdata)
                        .set_value(std::to_string(d).c_str());
                }
            }
        }
        // <layers/output>
        if ((node->get_output_size() > 0) &&
            !dynamic_cast<ngraph::op::Result*>(node.get())) {
            pugi::xml_node output = layer.append_child("output");
            for (auto o : node->outputs()) {
                pugi::xml_node port = output.append_child("port");
                port.append_attribute("id").set_value(port_id++);
                port.append_attribute("precision")
                    .set_value(get_output_precision_name(o).c_str());
                for (auto d : o.get_shape()) {
                    pugi::xml_node dim = port.append_child("dim");
                    dim.append_child(pugi::xml_node_type::node_pcdata)
                        .set_value(std::to_string(d).c_str());
                }
            }
        }
    }
    // <edges>
    std::vector<Edge> edge_mapping = create_edge_mapping(layer_ids, f);
    pugi::xml_node edges = netXml.append_child("edges");
    for (auto e : edge_mapping) {
        pugi::xml_node edge = edges.append_child("edge");
        edge.append_attribute("from-layer").set_value(e.from_layer);
        edge.append_attribute("from-port").set_value(e.from_port);
        edge.append_attribute("to-layer").set_value(e.to_layer);
        edge.append_attribute("to-port").set_value(e.to_port);
    }
}

}  // namespace

// ! [function_pass:serialize_cpp]
// serialize.cpp
bool pass::Serialize::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // prepare data
    pugi::xml_document xml_doc;
    std::vector<uint8_t> constants;
    switch (m_version) {
    case Version::IR_V10:
        ngfunction_2_irv10(xml_doc, constants, *f);
        break;
    default:
        NGRAPH_UNREACHABLE("");
        break;
    }

    // create xml file
    std::ofstream xml_file(m_xmlPath, std::ios::out);
    xml_doc.save(xml_file);

    // create bin file
    std::ofstream bin_file(m_binPath, std::ios::out | std::ios::binary);
    bin_file.write(reinterpret_cast<const char*>(constants.data()),
                   constants.size() * sizeof(constants[0]));

    // Return false because we didn't change nGraph Function
    return false;
}
// ! [function_pass:serialize_cpp]
