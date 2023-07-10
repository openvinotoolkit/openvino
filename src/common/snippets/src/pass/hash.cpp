// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/hash.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset1.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace ov {
namespace snippets {
namespace pass {

// helper
namespace {
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

template <typename T>
static uint64_t hash_combine(uint64_t seed, const T& a) {
    // Hash combine formula from boost
    return seed ^ (std::hash<T>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

void ovfunction_2_hash(uint64_t& hash,
                     const ov::Model& model);

namespace rt_info {

// some node attr is not type of ov::RuntimeAttribute, need dedicate visitor.
const std::vector<std::string> list_of_names{
    "PrimitivesPriority",
    "alt_width",
};

class NodeAuxRTInfoHasher {
public:
    explicit NodeAuxRTInfoHasher(uint64_t& hash) : m_hash(hash) {}

    void serialize(const ov::Node::RTMap& rt_info) {
        for (const auto& rt_info_name : list_of_names) {
            const auto& found_rt_info = rt_info.find(rt_info_name);
            if (found_rt_info != rt_info.end()) {
                std::stringstream strm;
                found_rt_info->second.print(strm);
                m_hash = hash_combine(m_hash, rt_info_name);
                m_hash = hash_combine(m_hash, strm.str());
            }
        }
    }

private:
    uint64_t& m_hash;
};

class RTInfoHasher : public ov::AttributeVisitor {
    uint64_t& m_rt_hash;

public:
    RTInfoHasher(uint64_t& rt_hash) : m_rt_hash(rt_hash) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
        } else {
            OPENVINO_THROW("Unsupported attribute type for serialization: ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        const auto& value = join(adapter.get());
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        const auto& value = join(adapter.get());
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        const auto& value = join(adapter.get());
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        const auto& value = join(adapter.get());
        m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW("Model type is unsupported for rt info serialization");
    }
};

} // namespace rt_info

class SnippetsHasher : public ov::AttributeVisitor {
    uint64_t& m_hash;
    const std::string& m_node_type_name;

    template <typename T>
    std::string create_atribute_list(ov::ValueAccessor<std::vector<T>>& adapter) {
        return join(adapter.get());
    }

public:
    SnippetsHasher(uint64_t& hash,
                  const std::string& node_type_name)
        : m_hash(hash),
          m_node_type_name(node_type_name) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (const auto& a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ngraph::Variable>>>(&adapter)) {
            m_hash = hash_combine(hash_combine(m_hash, name), a->get()->get_info().variable_id);
        } else if (const auto& a =
                       ov::as_type<ov::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter)) {
            if (name == "value" && m_node_type_name == "Constant") {
                m_hash = hash_combine(m_hash, std::string("Constant"));
                const int64_t size = a->get()->size();
                m_hash = hash_combine(hash_combine(m_hash, std::string("size")), size);
                auto data = static_cast<const char*>(a->get()->get_ptr());
                for (int64_t i = 0; i < size; i++) {
                    m_hash = hash_combine(m_hash, data[i]);
                }
            }
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
            const auto& attrs = a->get();
            // Update node attributes in data field
            for (const auto& attr : attrs) {
                m_hash = hash_combine(hash_combine(m_hash, attr.first), attr.second);
            }
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
            const auto& attrs = a->get();
            m_hash = hash_combine(hash_combine(m_hash, name), join(attrs));
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& attrs = a->get();
            auto shape_str = attrs.to_string();
            if (shape_str[0] == '[' && shape_str[shape_str.size() - 1] == ']')
                shape_str = shape_str.substr(1, shape_str.size() - 2);
            m_hash = hash_combine(hash_combine(m_hash, name), shape_str);
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            const auto& attrs = a->get();
            std::stringstream dim_str_stream;
            dim_str_stream << attrs;
            auto dim_str = dim_str_stream.str();
            if (dim_str[0] == '{' && dim_str[dim_str.size() - 1] == '}')
                dim_str = dim_str.substr(1, dim_str.size() - 2);
            m_hash = hash_combine(hash_combine(m_hash, name), dim_str);
        } else {
            OPENVINO_THROW("Unsupported attribute type for serialization: ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), static_cast<long long>(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_atribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_atribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_atribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_atribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_atribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        if (name == "net") {
            ovfunction_2_hash(m_hash, *adapter.get());
        } else {
            OPENVINO_THROW("Unsupported Model name.");
        }
    }
};

const std::unordered_map<ov::Node*, int> create_layer_ids(const ov::Model& model) {
    std::unordered_map<ov::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : model.get_ordered_ops()) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

const std::vector<Edge> create_edge_mapping(const std::unordered_map<ov::Node*, int>& layer_ids,
                                            const ov::Model& model) {
    std::vector<Edge> edges;
    for (const auto& node : model.get_ordered_ops()) {
        if (ov::op::util::is_parameter(node)) {
            continue;
        }

        for (const auto& i : node->inputs()) {
            auto source_output = i.get_source_output();
            auto source_node = source_output.get_node();
            auto current_node = i.get_node();

            OPENVINO_ASSERT(layer_ids.find(source_node) != layer_ids.end(), "Internal error");
            OPENVINO_ASSERT(layer_ids.find(current_node) != layer_ids.end(), "Internal error");

            Edge e{};
            e.from_layer = layer_ids.find(source_node)->second;
            e.from_port = static_cast<int>(source_node->get_input_size() + source_output.get_index());
            e.to_layer = layer_ids.find(current_node)->second;
            e.to_port = static_cast<int>(i.get_index());
            edges.push_back(e);
        }
    }
    std::sort(begin(edges), end(edges), [](const Edge& a, const Edge& b) -> bool {
        return a.from_layer < b.from_layer;
    });
    return edges;
}

std::string get_precision_name(const ov::element::Type& elem_type) {
    switch (elem_type) {
    case ::ov::element::Type_t::undefined:
    case ::ov::element::Type_t::dynamic:
        return "UNSPECIFIED";
    case ::ov::element::Type_t::f16:
        return "FP16";
    case ::ov::element::Type_t::f32:
        return "FP32";
    case ::ov::element::Type_t::bf16:
        return "BF16";
    case ::ov::element::Type_t::f64:
        return "FP64";
    case ::ov::element::Type_t::i4:
        return "I4";
    case ::ov::element::Type_t::i8:
        return "I8";
    case ::ov::element::Type_t::i16:
        return "I16";
    case ::ov::element::Type_t::i32:
        return "I32";
    case ::ov::element::Type_t::i64:
        return "I64";
    case ::ov::element::Type_t::u4:
        return "U4";
    case ::ov::element::Type_t::u8:
        return "U8";
    case ::ov::element::Type_t::u16:
        return "U16";
    case ::ov::element::Type_t::u32:
        return "U32";
    case ::ov::element::Type_t::u64:
        return "U64";
    case ::ov::element::Type_t::u1:
        return "BIN";
    case ::ov::element::Type_t::boolean:
        return "BOOL";
    default:
        OPENVINO_THROW("Unsupported precision: ", elem_type);
    }
}

void hash_rt_info(uint64_t& hash, const std::string& name, const ov::Any& data) {
    if (data.is<std::shared_ptr<ov::Meta>>()) {
        std::shared_ptr<ov::Meta> meta = data.as<std::shared_ptr<ov::Meta>>();
        ov::AnyMap& map = *meta;
        for (const auto& it : map) {
            hash_rt_info(hash, it.first, it.second);
        }
    } else if (data.is<ov::AnyMap>()) {
        const ov::AnyMap& any_map = data.as<ov::AnyMap>();
        for (const auto& it : any_map) {
            hash_rt_info(hash, it.first, it.second);
        }
    } else {
        std::string value = data.as<std::string>();
        hash = hash_combine(hash_combine(hash, std::string("value")), value);
    }
}

void ovfunction_2_hash(uint64_t& hash,
                     const ov::Model& model) {
    hash = hash_combine(hash, std::string("layers"));

    const std::unordered_map<ov::Node*, int> layer_ids = create_layer_ids(model);
    std::unordered_set<std::string> unique_names;

    auto sorted_ops = model.get_ordered_ops();

    // get_ordered_ops() returns operations after a topological sort. The topological sort reverses order of Parameters
    // and Results. So we need to put them into sorted_ops separately to ensure correct order of inputs and outputs.
    {
        std::vector<std::shared_ptr<ov::Node>> result;
        result.reserve(sorted_ops.size());
        for (const auto& param : model.get_parameters()) {
            result.emplace_back(param);
        }
        for (auto&& node : sorted_ops) {
            if (!ov::op::util::is_parameter(node) && !ov::op::util::is_output(node) && !ov::op::util::is_sink(node))
                result.emplace_back(node);
        }
        for (const auto& sink : model.get_sinks()) {
            result.emplace_back(sink);
        }
        for (const auto& res : model.get_results()) {
            result.emplace_back(res);
        }
        sorted_ops = result;
    }

    for (const auto& n : sorted_ops) {
        ov::Node* node = n.get();
        const std::string& node_type_name{node->get_type_name()};

        OPENVINO_ASSERT(layer_ids.find(node) != layer_ids.end(), "Internal error");
        // <layers>
        hash = hash_combine(hash, std::string("layer"));
        hash = hash_combine(hash_combine(hash, std::string("id")), layer_ids.find(node)->second);
        hash = hash_combine(hash_combine(hash, std::string("type")), node_type_name);

        // <layers/data> general attributes
        hash = hash_combine(hash, std::string("data"));
        auto append_runtime_info = [&](uint64_t& hash, ov::RTMap& attributes) {
            hash = hash_combine(hash, std::string("rt_info"));
            for (auto& item : attributes) {
                if (item.second.is<ov::RuntimeAttribute>()) {
                    auto& rt_attribute = item.second.as<ov::RuntimeAttribute>();
                    const auto& type_info = rt_attribute.get_type_info();
                    if (!strcmp(type_info.name, "fused_names")) {
                        continue;
                    }
                    hash = hash_combine(hash, std::string("attribute"));
                    hash = hash_combine(hash_combine(hash, std::string("name")), type_info.name);
                    hash = hash_combine(hash_combine(hash, std::string("version")), type_info.get_version());

                    rt_info::RTInfoHasher rt_info_visitor(hash);
                    rt_attribute.visit_attributes(rt_info_visitor);
                }
            }
        };

        append_runtime_info(hash, node->get_rt_info());

        int port_id = 0;
        // <layers/input>
        if (node->get_input_size() > 0) {
            hash = hash_combine(hash, std::string("input"));
            for (auto& i : node->inputs()) {
                hash = hash_combine(hash, std::string("port"));
                hash = hash_combine(hash_combine(hash, std::string("id")), port_id++);
                hash = hash_combine(hash_combine(hash, std::string("precision")), get_precision_name(i.get_element_type()));
                for (auto d : i.get_partial_shape()) {
                    hash = hash_combine(hash, std::string("dim"));
                    if (d.is_dynamic()) {
                        hash = hash_combine(hash, std::string("-1"));
                    } else {
                        hash = hash_combine(hash, std::to_string(d.get_length()));
                    }
                }
                append_runtime_info(hash, i.get_rt_info());
            }
        }
        // <layers/output>
        if ((node->get_output_size() > 0) && !ov::op::util::is_output(node)) {
            hash = hash_combine(hash, std::string("output"));
            // pugi::xml_node output = layer.append_child("output");
            for (auto& o : node->outputs()) {
                hash = hash_combine(hash, std::string("port"));
                hash = hash_combine(hash_combine(hash, std::string("id")), port_id++);
                hash = hash_combine(hash_combine(hash, std::string("precision")), get_precision_name(o.get_element_type()));

                for (auto d : o.get_partial_shape()) {
                    hash = hash_combine(hash, std::string("dim"));
                    if (d.is_dynamic()) {
                        hash = hash_combine(hash, std::string("-1"));
                    } else {
                        hash = hash_combine(hash, std::to_string(d.get_length()));
                    }
                }
                append_runtime_info(hash, o.get_rt_info());
            }
        }

        // fill <data> general attributes
        {
            SnippetsHasher visitor(hash, node_type_name);
            OPENVINO_ASSERT(node->visit_attributes(visitor), "Visitor API is not supported in ", node);
        }
        rt_info::NodeAuxRTInfoHasher{hash}.serialize(node->get_rt_info());
    }
    // <edges>
    const std::vector<Edge> edge_mapping = create_edge_mapping(layer_ids, model);
    hash = hash_combine(hash, std::string("edges"));
    for (auto e : edge_mapping) {
        hash = hash_combine(hash, std::string("edge"));
        hash = hash_combine(hash_combine(hash, std::string("from-layer")), e.from_layer);
        hash = hash_combine(hash_combine(hash, std::string("from-port")), e.from_port);
        hash = hash_combine(hash_combine(hash, std::string("to-layer")), e.to_layer);
        hash = hash_combine(hash_combine(hash, std::string("to-port")), e.to_port);
    }

    // Serialize rt info
    hash = hash_combine(hash, std::string("rt_info"));
    for (const auto& it : model.get_rt_info()) {
        hash_rt_info(hash, it.first, it.second);
    }
}

void snippets_to_hash(uint64_t& hash,
                   std::shared_ptr<ov::Model> model) {
    std::string name = "net";
    SnippetsHasher visitor(hash, name);
    visitor.on_attribute(name, model);
}

}  // namespace

bool Hash::run_on_model(const std::shared_ptr<ov::Model>& f) {
    uint64_t seed = 0;
    snippets_to_hash(seed, f);
    m_hash = seed;
    // Return false because we didn't change OpenVINO Model
    return false;
}

Hash::Hash(uint64_t& output_hash_value) : m_hash(output_hash_value) {}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
