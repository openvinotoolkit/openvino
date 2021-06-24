// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <ngraph/variant.hpp>
#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph_ops/framework_node.hpp"
#include "pugixml.hpp"
#include "transformations/serialize.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::Serialize, "Serialize", 0);

namespace {  // helpers
template <typename Container>
std::string join(const Container& c, const char* glue = ", ") {
    std::stringstream oss;
    const char* s = "";
    for (const auto& v : c) {
        oss << s << v;
        s = glue;
    }
    return oss.str();
}

struct Edge {
    int from_layer = 0;
    int from_port = 0;
    int to_layer = 0;
    int to_port = 0;
};

// Here operation type names are translated from ngraph convention to IR
// convention. Most of them are the same, but there are exceptions, e.g
// Constant (ngraph name) and Const (IR name). If there will be more
// discrepancies discovered, translations needs to be added here.
const std::unordered_map<std::string, std::string> translate_type_name_translator = {
    {"Constant", "Const"},
    {"PRelu", "PReLU"},
    {"Relu", "ReLU"},
    {"Softmax", "SoftMax"}};

std::string translate_type_name(const std::string& name) {
    auto found = translate_type_name_translator.find(name);
    if (found != end(translate_type_name_translator)) {
        return found->second;
    }
    return name;
}

size_t hash_combine(const void* v, int64_t size) {
    constexpr auto cel_size = sizeof(size_t);
    size_t seed = static_cast<size_t>(size);
    const auto data = static_cast<const size_t*>(v);
    const auto d_end = std::next(data, size / cel_size);
    // The constant value used as a magic number has been
    // traditionally used e.g. in boost library's hash_combine.
    // It happens to be derived from the golden ratio.
    for (auto d = data; d != d_end; ++d) {
        seed ^= *d + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    size_t last_bytes{0};
    std::memcpy(&last_bytes, d_end, size % cel_size);
    seed ^= last_bytes + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

class ConstantWriter {
public:
    using FilePosition = int64_t;
    using HashValue = size_t;
    using ConstWritePositions = std::unordered_map<HashValue, std::pair<FilePosition, void const *>>;

    ConstantWriter(std::ostream& bin_data, bool enable_compression = true)
        : m_binary_output(bin_data)
        , m_enable_compression(enable_compression) {
    }

    FilePosition write(const char* ptr, size_t size) {
        const auto offset = m_binary_output.tellp();
        if (!m_enable_compression) {
            m_binary_output.write(ptr, size);
            return offset;
        }
        // This hash is weak (but efficient) and must be replace with some other
        // more stable hash algorithm. For example current hash algorithms gives
        // the same hash for {2, 2} and {0, 128} arrays. So we have to compare
        // values when finding a match in hash map.
        const HashValue hash = hash_combine(ptr, size);
        const auto found = m_hash_to_file_positions.find(hash);
        if (found != end(m_hash_to_file_positions) &&
            memcmp(static_cast<void const*>(ptr), found->second.second, size) == 0) {
            return found->second.first;
        }

        m_binary_output.write(ptr, size);
        m_hash_to_file_positions.insert({hash, {offset, static_cast<void const *>(ptr)}});

        return offset;
    }

private:
    ConstWritePositions m_hash_to_file_positions;
    std::ostream& m_binary_output;
    bool m_enable_compression;
};

void ngfunction_2_irv10(pugi::xml_node& node,
                        const ngraph::Function& f,
                        const std::map<std::string, ngraph::OpSet>& custom_opsets,
                        ConstantWriter& constant_write_handler);

// Some of the operators were added to wrong opsets. This is a mapping
// that allows such operators to be serialized with proper opsets.
// If new operators are discovered that have the same problem, the mapping
// needs to be updated here. The keys contain op name and version in NodeTypeInfo.
const std::unordered_map<ngraph::Node::type_info_t, std::string>
    special_operator_to_opset_assignments = {{ngraph::Node::type_info_t("ShuffleChannels", 0), "opset3"}};

std::string get_special_opset_for_op(const ngraph::Node::type_info_t& type_info) {
    auto found = special_operator_to_opset_assignments.find(type_info);
    if (found != end(special_operator_to_opset_assignments)) {
        return found->second;
    }
    return "";
}

namespace rt_info {
const std::vector<std::string> list_of_names {
    "PrimitivesPriority",
    "alt_width",
};

class XmlSerializer {
public:
    explicit XmlSerializer(pugi::xml_node &xml_node)
        : m_xml_node(xml_node) {
    }

    void serialize(const ngraph::Node::RTMap& rt_info) {
        for (const auto& rt_info_name : list_of_names) {
            const auto &found_rt_info = rt_info.find(rt_info_name);
            if (found_rt_info != rt_info.end()) {
                xml_node_append_attribute<std::string>(rt_info_name, found_rt_info->second);
            }
        }
    }

private:
    template<typename VariantType>
    void xml_node_append_attribute(const std::string& name,
                                   const std::shared_ptr<ngraph::Variant>& variant) {
        if ( auto v = std::dynamic_pointer_cast<ngraph::VariantImpl<VariantType>>(variant) ) {
            const auto& value = v->get();
            m_xml_node.append_attribute(name.c_str()).set_value(value.c_str());
        }
    }

    pugi::xml_node& m_xml_node;
};

} // namespace rt_info

class XmlSerializer : public ngraph::AttributeVisitor {
    pugi::xml_node& m_xml_node;
    const std::string& m_node_type_name;
    const std::map<std::string, ngraph::OpSet>& m_custom_opsets;
    ConstantWriter& m_constant_write_handler;

    template <typename T>
    std::string create_atribute_list(
        ngraph::ValueAccessor<std::vector<T>>& adapter) {
        return join(adapter.get());
    }

    std::vector<std::string> map_type_from_body(const pugi::xml_node& xml_node,
        const std::string& map_type) {
        std::vector<std::string> output;
        for (pugi::xml_node node : xml_node.child("body").child("layers")) {
            if (!map_type.compare(node.attribute("type").value())) {
                output.push_back(node.attribute("id").value());
            }
        }

        // ops for serialized body function are provided in reversed order
        std::reverse(output.begin(), output.end());

        return output;
    }

    void input_descriptions_on_adapter(const std::vector<std::shared_ptr<
                                        ngraph::op::util::SubGraphOp::InputDescription>>& input_descriptions,
                                        const std::vector<std::string>& parameter_mapping,
                                        const std::vector<std::string>& result_mapping,
                                        pugi::xml_node& port_map) {
        NGRAPH_CHECK(!parameter_mapping.empty(), "No parameters found in body Function.");

        if (!m_xml_node.parent().child("port_map")) {
            port_map = m_xml_node.parent().insert_child_before("port_map", m_xml_node.parent().first_child());
        }

        for (const auto& input_description : input_descriptions) {
            pugi::xml_node input = port_map.append_child("input");
            input.append_attribute("external_port_id").set_value(input_description->m_input_index);
            input.append_attribute("internal_layer_id").set_value(parameter_mapping[input_description->m_body_parameter_index].c_str());

            if (auto slice_input = as_type_ptr<ngraph::op::util::SubGraphOp::SliceInputDescription>(input_description)) {
                input.prepend_attribute("axis").set_value(slice_input->m_axis);
                input.append_attribute("start").set_value(slice_input->m_start);
                input.append_attribute("end").set_value(slice_input->m_end);
                input.append_attribute("stride").set_value(slice_input->m_stride);
                input.append_attribute("part_size").set_value(slice_input->m_part_size);
            } else if (auto merged_input = as_type_ptr<ngraph::op::util::SubGraphOp::MergedInputDescription>(input_description)) {
                pugi::xml_node back_edges = m_xml_node.parent().child("back_edges");
                if (!back_edges) {
                    back_edges = m_xml_node.parent().insert_child_after("back_edges", port_map);
                }
                pugi::xml_node edge = back_edges.append_child("edge");
                edge.append_attribute("from-layer").set_value(result_mapping[merged_input->m_body_value_index].c_str());
                edge.append_attribute("to-layer").set_value(parameter_mapping[merged_input->m_body_parameter_index].c_str());
            }
        }
    }

    void output_descriptions_on_adapter(const std::vector<std::shared_ptr<
                                        ngraph::op::util::SubGraphOp::OutputDescription>>& output_descriptions,
                                        const uint32_t& input_count,
                                        const std::vector<std::string>& result_mapping,
                                        pugi::xml_node& port_map) {
        NGRAPH_CHECK(!result_mapping.empty(), "No results found in body Function.");

        if (!port_map) {
            port_map = m_xml_node.parent().insert_child_before("port_map", m_xml_node.parent().first_child());
        }

        for (const auto& output_description : output_descriptions) {
            pugi::xml_node output = port_map.append_child("output");
            output.append_attribute("external_port_id").set_value(input_count + output_description->m_output_index);
            output.append_attribute("internal_layer_id").set_value(result_mapping[output_description->m_body_value_index].c_str());

            if (auto concat_output = as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(output_description)) {
                output.prepend_attribute("axis").set_value(concat_output->m_axis);
                output.append_attribute("start").set_value(concat_output->m_start);
                output.append_attribute("end").set_value(concat_output->m_end);
                output.append_attribute("stride").set_value(concat_output->m_stride);
                output.append_attribute("part_size").set_value(concat_output->m_part_size);
            }
        }
    }

    void special_body_ports_on_adapter(const ngraph::op::v5::Loop::SpecialBodyPorts& special_body_ports,
                                    const std::vector<std::string>& parameter_mapping,
                                    const std::vector<std::string>& result_mapping,
                                    pugi::xml_node& port_map) {
        NGRAPH_CHECK(port_map, "port_map section not found, purpose attribute cannot be added.");

        if (special_body_ports.current_iteration_input_idx != -1) {
            pugi::xml_node iter_input = port_map.append_child("input");
            iter_input.append_attribute("external_port_id").set_value("-1");
            iter_input.append_attribute("internal_layer_id").set_value(parameter_mapping[special_body_ports.current_iteration_input_idx].c_str());
            iter_input.append_attribute("purpose").set_value("current_iteration");
        }

        if (special_body_ports.body_condition_output_idx != -1) {
            pugi::xml_node exec_output = port_map.append_child("output");
            exec_output.append_attribute("external_port_id").set_value("-1");
            exec_output.append_attribute("internal_layer_id").set_value(result_mapping[special_body_ports.body_condition_output_idx].c_str());
            exec_output.append_attribute("purpose").set_value("execution_condition");
        }
    }

public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  const std::map<std::string, ngraph::OpSet>& custom_opsets,
                  ConstantWriter& constant_write_handler)
        : m_xml_node(data)
        , m_node_type_name(node_type_name)
        , m_custom_opsets(custom_opsets)
        , m_constant_write_handler(constant_write_handler) {
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        if (m_xml_node.parent().child("body")) {
            std::vector<std::string> result_mapping = map_type_from_body(m_xml_node.parent(), "Result");
            std::vector<std::string> parameter_mapping = map_type_from_body(m_xml_node.parent(), "Parameter");
            pugi::xml_node port_map = m_xml_node.parent().child("port_map");

            NGRAPH_CHECK(!parameter_mapping.empty() || !result_mapping.empty(), "No parameters or results found in body Function.");
            // TI, Loop do not have attributtes as regular ops, it is necessary to append "port_map" and
            // "back_edges" to layer above (m_xml_node.parent()) as in ngfunction_2_irv10() layer (here "m_xml_node")
            // with empty attributes is removed.
            if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<std::shared_ptr
                                <ngraph::op::util::SubGraphOp::InputDescription>>>>(&adapter)) {
                input_descriptions_on_adapter(a->get(), parameter_mapping, result_mapping, port_map);
            } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<std::shared_ptr
                                <ngraph::op::util::SubGraphOp::OutputDescription>>>>(&adapter)) {
                uint32_t op_input_count = 0;
                for (auto c = m_xml_node.parent().child("input").first_child(); !c.empty(); c = c.next_sibling()) {
                    op_input_count++;
                }
                output_descriptions_on_adapter(a->get(), op_input_count, result_mapping, port_map);
            } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::v5::Loop::SpecialBodyPorts>>(&adapter)) {
                special_body_ports_on_adapter(a->get(), parameter_mapping, result_mapping, port_map);
            }
        } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::Variable>>>(&adapter)) {
                m_xml_node.append_attribute(name.c_str()).set_value(a->get()->get_info().variable_id.c_str());
        } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter)) {
            if (name == "value" &&  translate_type_name(m_node_type_name) == "Const") {
                const int64_t size = a->get()->size();
                int64_t offset = m_constant_write_handler.write(
                    static_cast<const char *>(a->get()->get_ptr()), size);

                m_xml_node.append_attribute("offset").set_value(offset);
                m_xml_node.append_attribute("size").set_value(size);
            }
        } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<op::FrameworkNodeAttrs>>(&adapter)) {
            const auto & attrs = a->get();

            // Update type and version attributes
            pugi::xml_node layer = m_xml_node.parent();

            auto type_attr = layer.attribute("type");
            auto version_attr = layer.attribute("version");

            type_attr.set_value(attrs.get_type_name().c_str());

            if (!attrs.get_opset_name().empty()) {
                version_attr.set_value(attrs.get_opset_name().c_str());
            } else {
                layer.remove_attribute("version");
            }

            // Update node attributes in data field
            for (const auto & attr : attrs) {
                m_xml_node.append_attribute(attr.first.c_str()).set_value(attr.second.c_str());
            }
        } else {
            throw ngraph_error("Unsupported attribute type for serialization: " + name);
        }
    }

    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<bool>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<std::string>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(adapter.get().c_str());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<int64_t>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<double>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<int>>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_xml_node.append_attribute(name.c_str())
            .set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::shared_ptr<Function>>& adapter) override {
        if (name == "body") {
            // TI, Loop do not have attributtes as regular ops, it is necessary to append "body"
            // to layer above (m_xml_node.parent()) as in ngfunction_2_irv10() layer (m_xml_node) with empty attributes
            // is removed.
            pugi::xml_node xml_body = m_xml_node.parent().append_child(name.c_str());
            ngfunction_2_irv10(xml_body, *adapter.get(), m_custom_opsets, m_constant_write_handler);
            xml_body.remove_attribute("name");
            xml_body.remove_attribute("version");
        } else if (name == "net") {
            ngfunction_2_irv10(m_xml_node, *adapter.get(), m_custom_opsets, m_constant_write_handler);
        } else {
            NGRAPH_CHECK(false, "Unsupported Function name.");
        }
    }
};

const std::unordered_map<ngraph::Node*, int> create_layer_ids(
    const ngraph::Function& f) {
    std::unordered_map<ngraph::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : f.get_ordered_ops()) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

const std::vector<Edge> create_edge_mapping(
    const std::unordered_map<ngraph::Node*, int>& layer_ids,
    const ngraph::Function& f) {
    std::vector<Edge> edges;
    for (const auto& node : f.get_ordered_ops()) {
        if (ngraph::op::is_parameter(node)) {
            continue;
        }

        for (const auto& i : node->inputs()) {
            auto source_output = i.get_source_output();
            auto source_node = source_output.get_node();
            auto current_node = i.get_node();

            NGRAPH_CHECK(layer_ids.find(source_node) != layer_ids.end(),
                         "Internal error");
            NGRAPH_CHECK(layer_ids.find(current_node) != layer_ids.end(),
                         "Internal error");

            Edge e{};
            e.from_layer = layer_ids.find(source_node)->second;
            e.from_port =
                source_node->get_input_size() + source_output.get_index();
            e.to_layer = layer_ids.find(current_node)->second;
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

std::string get_opset_name(
    const ngraph::Node* n,
    const std::map<std::string, ngraph::OpSet>& custom_opsets) {
    auto opsets = std::array<std::reference_wrapper<const ngraph::OpSet>, 7>{
        ngraph::get_opset1(), ngraph::get_opset2(), ngraph::get_opset3(),
        ngraph::get_opset4(), ngraph::get_opset5(), ngraph::get_opset6(),
        ngraph::get_opset7()};

    auto special_opset = get_special_opset_for_op(n->get_type_info());
    if (!special_opset.empty()) {
        return special_opset;
    }
    // return the oldest opset name where node type is present
    for (size_t idx = 0; idx < opsets.size(); idx++) {
        if (opsets[idx].get().contains_op_type(n)) {
            return "opset" + std::to_string(idx + 1);
        }
    }

    for (const auto& custom_opset : custom_opsets) {
        std::string name = custom_opset.first;
        ngraph::OpSet opset = custom_opset.second;
        if (opset.contains_op_type(n)) {
            return name;
        }
    }

    return "experimental";
}

std::string get_precision_name(const ngraph::element::Type & elem_type) {
    switch (elem_type) {
    case ::ngraph::element::Type_t::undefined:
        return "UNSPECIFIED";
    case ::ngraph::element::Type_t::f16:
        return "FP16";
    case ::ngraph::element::Type_t::f32:
        return "FP32";
    case ::ngraph::element::Type_t::bf16:
        return "BF16";
    case ::ngraph::element::Type_t::f64:
        return "FP64";
    case ::ngraph::element::Type_t::i4:
        return "I4";
    case ::ngraph::element::Type_t::i8:
        return "I8";
    case ::ngraph::element::Type_t::i16:
        return "I16";
    case ::ngraph::element::Type_t::i32:
        return "I32";
    case ::ngraph::element::Type_t::i64:
        return "I64";
    case ::ngraph::element::Type_t::u4:
        return "U4";
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
        std::stringstream msg;
        msg << "Unsupported precision: " << elem_type;
        throw ngraph_error(msg.str());
    }
}

std::string escape_delim(const std::string& name, const char delim = ',') {
    std::string result_name = name;
    const std::string escaped_delim = std::string("\\") + delim;
    size_t index = result_name.find(delim, 0);
    while (index != std::string::npos) {
        result_name.replace(index, 1, escaped_delim);
        index = result_name.find(delim, index + 2);
    }
    return result_name;
}

std::string generate_unique_name(
    const std::unordered_set<std::string>& unique_names, std::string base_name,
    int suffix) {
    std::string new_name = base_name + std::to_string(suffix);
    if (unique_names.find(new_name) == unique_names.end()) {
        return new_name;
    } else {
        suffix++;
        return generate_unique_name(unique_names, base_name, suffix);
    }
}

// TODO: remove when CNNNetwork will be supporting not-unique names
std::string get_node_unique_name(std::unordered_set<std::string>& unique_names,
                                 const ngraph::Node* n) {
    std::string name = n->get_friendly_name();
    if (unique_names.find(name) != unique_names.end()) {
        name = generate_unique_name(unique_names, name, 0);
    }
    unique_names.insert(name);
    return name;
}

void visit_exec_graph_node(pugi::xml_node& layer, const ngraph::Node* n) {
    auto data = layer.child("data");
    for (const auto& param : n->get_rt_info()) {
        if (auto variant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(param.second)) {
            const std::string & name = param.first;
            const std::string & value = variant->get();

            if (name == "layerType") {
                layer.attribute("type").set_value(value.c_str());
                continue;
            }

            data.append_attribute(name.c_str()).set_value(value.c_str());
        }
    }
}

bool is_exec_graph(const ngraph::Function& f) {
    // go over all operations and check whether performance stat is set
    for (const auto& op : f.get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        if (rtInfo.find("execTimeMcs") != rtInfo.end()) {
            return true;
        }
    }
    return false;
}

bool has_dynamic_output(std::shared_ptr<Node> n) {
    for (size_t i = 0; i < n->get_output_size(); i++) {
        if (n->get_output_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

bool resolve_dynamic_shapes(const ngraph::Function& f) {
    const auto & f_ops = f.get_ordered_ops();
    if (std::all_of(f_ops.begin(), f_ops.end(),
            [](std::shared_ptr<Node> results) {
                return !results->is_dynamic() && !has_dynamic_output(results); })) {
        return false;
    }

    auto f_clone = ngraph::clone_function(f);
    const auto & f_clone_ops = f_clone->get_ordered_ops();
    NGRAPH_CHECK(f_ops.size() == f_clone_ops.size(), "Unexpected get_ordered_ops method behaviour");

    for (size_t id = 0; id < f_ops.size(); ++id) {
        auto & op = f_ops[id];
        auto & clone_op = f_clone_ops[id];

        if (auto op_subgraph = std::dynamic_pointer_cast<op::util::SubGraphOp>(op)) {
            resolve_dynamic_shapes(*op_subgraph->get_function());
        }

        op->validate_and_infer_types();
        clone_op->validate_and_infer_types();

        // dynamic_to_static function converts dynamic dimensions to static using
        // upperbound (get_max_length) dimension value.
        auto dynamic_to_static = [&op](const PartialShape & shape) -> PartialShape {
            if (shape.is_static() || shape.rank().is_dynamic()) {
                return shape;
            }
            std::vector<Dimension> out_shape;
            std::transform(std::begin(shape), std::end(shape),
                           std::back_inserter(out_shape),
                           [](const Dimension& d) -> Dimension {
                               return d.get_max_length();
                           });
            NGRAPH_CHECK(PartialShape(out_shape).is_static(),
                         "Dynamic dimension cannot be resolved in ", op);
            return out_shape;
        };

        OutputVector replacements(clone_op->get_output_size());
        if (!clone_op->constant_fold(replacements, clone_op->input_values())) {
            for (size_t output_id = 0; output_id < clone_op->get_output_size(); ++output_id) {
                clone_op->set_output_type(output_id, clone_op->output(output_id).get_element_type(),
                        dynamic_to_static(clone_op->output(output_id).get_partial_shape()));
                op->set_output_type(output_id, clone_op->output(output_id).get_element_type(),
                        clone_op->output(output_id).get_partial_shape());
            }
        } else {
            for (size_t output_id = 0; output_id < clone_op->get_output_size(); ++output_id) {
                op->set_output_type(output_id, replacements[output_id].get_element_type(),
                        replacements[output_id].get_partial_shape());
            }

            for (size_t i = 0; i < replacements.size(); ++i) {
                auto node_output = clone_op->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
                    node_output.replace(replacement);
                }
            }
        }
    }
    return true;
}

void ngfunction_2_irv10(pugi::xml_node& netXml,
                        const ngraph::Function& f,
                        const std::map<std::string, ngraph::OpSet>& custom_opsets,
                        ConstantWriter& constant_node_write_handler) {
    netXml.append_attribute("name").set_value(f.get_friendly_name().c_str());
    netXml.append_attribute("version").set_value("10");
    pugi::xml_node layers = netXml.append_child("layers");

    const std::unordered_map<ngraph::Node*, int> layer_ids =
        create_layer_ids(f);
    std::unordered_set<std::string> unique_names;

    bool has_dynamic_shapes = resolve_dynamic_shapes(f);

    const bool exec_graph = is_exec_graph(f);

    for (const auto& n : f.get_ordered_ops()) {
        ngraph::Node* node = n.get();
        const std::string & node_type_name{node->get_type_name()};

        NGRAPH_CHECK(layer_ids.find(node) != layer_ids.end(), "Internal error");
        // <layers>
        pugi::xml_node layer = layers.append_child("layer");
        layer.append_attribute("id").set_value(layer_ids.find(node)->second);
        layer.append_attribute("name").set_value(get_node_unique_name(unique_names, node).c_str());
        layer.append_attribute("type").set_value(translate_type_name(node_type_name).c_str());
        if (!exec_graph) {
            layer.append_attribute("version").set_value(get_opset_name(node, custom_opsets).c_str());
        }

        // <layers/data> general attributes
        pugi::xml_node data = layer.append_child("data");

        int port_id = 0;
        // <layers/input>
        if (node->get_input_size() > 0) {
            pugi::xml_node input = layer.append_child("input");
            for (const auto & i : node->inputs()) {
                NGRAPH_CHECK(i.get_partial_shape().is_static(),
                             "Unsupported dynamic input shape in ", node);

                // WA for LSTMCellv0, peephole input shall not be serialized
                if (i.get_index() == 6 && dynamic_cast<opset1::LSTMCell *>(node)) {
                    port_id++;
                    continue;
                }

                pugi::xml_node port = input.append_child("port");
                port.append_attribute("id").set_value(port_id++);
                port.append_attribute("precision")
                        .set_value(get_precision_name(i.get_element_type()).c_str());
                for (auto d : i.get_shape()) {
                    pugi::xml_node dim = port.append_child("dim");
                    dim.append_child(pugi::xml_node_type::node_pcdata)
                        .set_value(std::to_string(d).c_str());
                }
            }

            if (node_type_name == "TensorIterator" || node_type_name == "Loop") {
                layer.prepend_move(input);
            }
        }
        // <layers/output>
        if ((node->get_output_size() > 0) && !ngraph::op::is_output(node)) {
            pugi::xml_node output = layer.append_child("output");
            for (const auto & o : node->outputs()) {
                NGRAPH_CHECK(o.get_partial_shape().is_static(),
                             "Unsupported dynamic output shape in ", node);

                pugi::xml_node port = output.append_child("port");
                port.append_attribute("id").set_value(port_id++);
                port.append_attribute("precision")
                    .set_value(get_precision_name(o.get_element_type()).c_str());

                // Sort tensor names
                const auto & tensor_names = o.get_tensor().get_names();
                std::vector<std::string> vector_names(tensor_names.begin(), tensor_names.end());
                sort(vector_names.begin(), vector_names.end());

                std::string names;
                for (const auto& name : vector_names) {
                    if (!names.empty())
                        names += ",";
                    names += escape_delim(name);
                }
                if (!names.empty()) {
                    port.append_attribute("names").set_value(names.c_str());
                }

                for (auto d : o.get_shape()) {
                    pugi::xml_node dim = port.append_child("dim");
                    dim.append_child(pugi::xml_node_type::node_pcdata)
                        .set_value(std::to_string(d).c_str());
                }
            }
            if (node_type_name == "TensorIterator" || node_type_name == "Loop") {
                layer.insert_move_after(output, layer.first_child());
            }
        }

        // fill <data> general attributes
        XmlSerializer visitor(data, node_type_name, custom_opsets, constant_node_write_handler);
        NGRAPH_CHECK(node->visit_attributes(visitor), "Visitor API is not supported in ", node);
        rt_info::XmlSerializer{data}.serialize(node->get_rt_info());

        if (exec_graph) {
            visit_exec_graph_node(layer, node);
        }

        const bool data_attr_size =
            data.attributes().begin() == data.attributes().end();
        if (data_attr_size) {
            layer.remove_child(data);
        }
    }
    // <edges>
    const std::vector<Edge> edge_mapping = create_edge_mapping(layer_ids, f);
    pugi::xml_node edges = netXml.append_child("edges");
    for (auto e : edge_mapping) {
        // WA for LSTMCellv0, peephole input shall not be serialized
        if (e.to_port == 6) {
            auto type_info = f.get_ordered_ops()[e.to_layer]->get_type_info();
            if (!strcmp(type_info.name, "LSTMCell") && type_info.version == 0) {
                continue;
            }
        }
        pugi::xml_node edge = edges.append_child("edge");
        edge.append_attribute("from-layer").set_value(e.from_layer);
        edge.append_attribute("from-port").set_value(e.from_port);
        edge.append_attribute("to-layer").set_value(e.to_layer);
        edge.append_attribute("to-port").set_value(e.to_port);
    }
    // move back dynamic shapes
    if (has_dynamic_shapes) {
        f.validate_nodes_and_infer_types();
    }
}
}  // namespace

// ! [function_pass:serialize_cpp]
// serialize.cpp
bool pass::Serialize::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(Serialize);

    auto serializeFunc = [&] (std::ostream & xml_file, std::ostream & bin_file) {
        switch (m_version) {
        case Version::IR_V10:
            {
                std::string name = "net";
                pugi::xml_document xml_doc;
                pugi::xml_node net_node = xml_doc.append_child(name.c_str());
                ConstantWriter constant_write_handler(bin_file);
                XmlSerializer visitor(net_node, name, m_custom_opsets, constant_write_handler);
                visitor.on_attribute(name, f);

                xml_doc.save(xml_file);
                xml_file.flush();
                bin_file.flush();
            }
            break;
        default:
            NGRAPH_UNREACHABLE("Unsupported version");
            break;
        }
    };

    if (m_xmlFile && m_binFile) {
        serializeFunc(*m_xmlFile, *m_binFile);
    } else {
        std::ofstream bin_file(m_binPath, std::ios::out | std::ios::binary);
        NGRAPH_CHECK(bin_file, "Can't open bin file: \"" + m_binPath + "\"");

        // create xml file
        std::ofstream xml_file(m_xmlPath, std::ios::out);
        NGRAPH_CHECK(xml_file, "Can't open xml file: \"" + m_xmlPath + "\"");

        try {
            serializeFunc(xml_file, bin_file);
        } catch (const ngraph::CheckFailure& e) {
            // optimization decission was made to create .bin file upfront and
            // write to it directly instead of buffering its content in memory,
            // hence we need to delete it here in case of failure
            xml_file.close();
            bin_file.close();
            std::remove(m_xmlPath.c_str());
            std::remove(m_binPath.c_str());
            throw;
        }
    }

    // Return false because we didn't change nGraph Function
    return false;
}

namespace {

std::string valid_xml_path(const std::string &path) {
    NGRAPH_CHECK(path.length() > 4, "Path for xml file is to short: \"" + path + "\"");

    const char *const extension = ".xml";
    const bool has_xml_extension = path.rfind(extension) == path.size() - std::strlen(extension);
    NGRAPH_CHECK(has_xml_extension,
                 "Path for xml file doesn't contains file name with 'xml' extension: \"" +
                     path + "\"");
    return path;
}

std::string provide_bin_path(const std::string &xmlPath, const std::string &binPath) {
    if (!binPath.empty()) {
        return binPath;
    }
    assert(xmlPath.size() > 4); // should be check by valid_xml_path
    std::string bestPath = xmlPath;
    const char *const extension = "bin";
    const auto ext_size = std::strlen(extension);
    bestPath.replace(bestPath.size() - ext_size, ext_size, extension);
    return bestPath;
}

} // namespace

pass::Serialize::Serialize(std::ostream& xmlFile,
                           std::ostream& binFile,
                           pass::Serialize::Version version,
                           std::map<std::string, OpSet> custom_opsets)
    : m_xmlFile{&xmlFile}
    , m_binFile{&binFile}
    , m_xmlPath{}
    , m_binPath{}
    , m_version{version}
    , m_custom_opsets{custom_opsets}
{
}

pass::Serialize::Serialize(const std::string& xmlPath,
                           const std::string& binPath,
                           pass::Serialize::Version version,
                           std::map<std::string, OpSet> custom_opsets)
    : m_xmlFile{nullptr}
    , m_binFile{nullptr}
    , m_xmlPath{valid_xml_path(xmlPath)}
    , m_binPath{provide_bin_path(xmlPath, binPath)}
    , m_version{version}
    , m_custom_opsets{custom_opsets}
{
}
// ! [function_pass:serialize_cpp]
