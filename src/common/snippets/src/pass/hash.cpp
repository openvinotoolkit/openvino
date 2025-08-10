// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/hash.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov::snippets::pass {

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

enum class AttrType : uint8_t {
    layers,
    layer,
    id,
    type,
    data,
    rt_info,
    attribute,
    name,
    version,
    input,
    port,
    precision,
    dimension,
    output,
    value,
    edges,
    edge,
    from_layer,
    from_port,
    to_layer,
    to_port,
    constant,
    size
};

template <typename T, std::enable_if_t<!std::is_enum_v<T>, int> = 0>
uint64_t hash_combine(uint64_t seed, const T& v) {
    // Hash combine formula from boost
    return seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
uint64_t hash_combine(uint64_t seed, const T& v) {
    using underlying_t = std::underlying_type_t<T>;
    return hash_combine(seed, static_cast<underlying_t>(v));
}

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
    explicit RTInfoHasher(uint64_t& rt_hash) : m_rt_hash(rt_hash) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto* a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            m_rt_hash = hash_combine(hash_combine(m_rt_hash, name), value);
        } else {
            OPENVINO_THROW("Unsupported attribute type for snippets hash generation: ", name);
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

    void on_adapter([[maybe_unused]] const std::string& name,
                    [[maybe_unused]] ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW("Model type is unsupported for snippets rt info hash generation");
    }
};

}  // namespace rt_info

void ovfunction_2_hash(uint64_t& hash, const ov::Model& model);

class SnippetsHasher : public ov::AttributeVisitor {
    uint64_t& m_hash;
    const std::string& m_node_type_name;

    template <typename T>
    std::string create_attribute_list(ov::ValueAccessor<std::vector<T>>& adapter) {
        return join(adapter.get());
    }

public:
    SnippetsHasher(uint64_t& hash, const std::string& node_type_name)
        : m_hash(hash),
          m_node_type_name(node_type_name) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (const auto& a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            m_hash = hash_combine(hash_combine(m_hash, name), a->get()->get_info().variable_id);
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&adapter)) {
            if (name == "value" && m_node_type_name == "Constant") {
                m_hash = hash_combine(m_hash, AttrType::constant);
                const int64_t size = a->get()->size();
                m_hash = hash_combine(hash_combine(m_hash, AttrType::size), size);
                const auto* data = static_cast<const char*>(a->get()->get_ptr());
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
            m_hash = hash_combine(hash_combine(m_hash, name), shape_str);
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            const auto& attrs = a->get();
            std::stringstream dim_str_stream;
            dim_str_stream << attrs;
            auto dim_str = dim_str_stream.str();
            m_hash = hash_combine(hash_combine(m_hash, name), dim_str);
        } else {
            OPENVINO_THROW("Unsupported attribute type for snippets hash generation: ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), static_cast<int64_t>(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_attribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_attribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_attribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_attribute_list(adapter));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_hash = hash_combine(hash_combine(m_hash, name), create_attribute_list(adapter));
    }
    void on_adapter([[maybe_unused]] const std::string& name,
                    ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        ovfunction_2_hash(m_hash, *adapter.get());
    }
};

std::unordered_map<ov::Node*, int> create_layer_ids(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops) {
    std::unordered_map<ov::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : ordered_ops) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

std::vector<Edge> create_edge_mapping(const std::unordered_map<ov::Node*, int>& layer_ids,
                                      const std::vector<std::shared_ptr<ov::Node>>& ordered_ops) {
    std::vector<Edge> edges;
    for (const auto& node : ordered_ops) {
        if (ov::op::util::is_parameter(node)) {
            continue;
        }

        for (const auto& i : node->inputs()) {
            auto source_output = i.get_source_output();
            auto* source_node = source_output.get_node();
            auto* current_node = i.get_node();

            if (layer_ids.find(source_node) == layer_ids.end() || layer_ids.find(current_node) == layer_ids.end()) {
                OPENVINO_THROW("Failed creat edge map in snippets hash generation");
            }

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

void hash_rt_info(uint64_t& hash, const ov::Any& data) {
    if (data.is<std::shared_ptr<ov::Meta>>()) {
        const auto& meta = data.as<std::shared_ptr<ov::Meta>>();
        ov::AnyMap& map = *meta;
        for (const auto& it : map) {
            hash_rt_info(hash, it.second);
        }
    } else if (data.is<ov::AnyMap>()) {
        const auto& any_map = data.as<ov::AnyMap>();
        for (const auto& it : any_map) {
            hash_rt_info(hash, it.second);
        }
    } else {
        const auto& value = data.as<std::string>();
        hash = hash_combine(hash_combine(hash, AttrType::value), value);
    }
}

void ovfunction_2_hash(uint64_t& hash, const ov::Model& model) {
    hash = hash_combine(hash, AttrType::layers);

    auto ordered_ops = model.get_ordered_ops();
    const std::unordered_map<ov::Node*, int> layer_ids = create_layer_ids(ordered_ops);
    std::unordered_set<std::string> unique_names;

    for (const auto& n : ordered_ops) {
        ov::Node* node = n.get();
        const std::string& node_type_name{node->get_type_name()};

        if (layer_ids.find(node) == layer_ids.end()) {
            OPENVINO_THROW("Failed to find layer's id in snippets hash generation.");
        }
        // <layers>
        hash = hash_combine(hash, AttrType::layer);
        hash = hash_combine(hash_combine(hash, AttrType::id), layer_ids.find(node)->second);
        hash = hash_combine(hash_combine(hash, AttrType::type), node_type_name);

        // <layers/data> general attributes
        hash = hash_combine(hash, AttrType::data);
        auto append_runtime_info = [&](uint64_t& hash, ov::RTMap& attributes) {
            hash = hash_combine(hash, AttrType::rt_info);
            for (const auto& [name, attribute] : attributes) {
                if (attribute.is<ov::RuntimeAttribute>()) {
                    if (const auto& rt_attribute = attribute.as<ov::RuntimeAttribute>();
                        rt_attribute.is_deterministic()) {
                        const auto& type_info = rt_attribute.get_type_info();
                        hash = hash_combine(hash, AttrType::attribute);
                        hash = hash_combine(hash_combine(hash, AttrType::name), type_info.name);
                        hash = hash_combine(hash_combine(hash, AttrType::version), type_info.get_version());

                        rt_info::RTInfoHasher rt_info_visitor(hash);
                        rt_attribute.visit_attributes(rt_info_visitor);
                    }
                }
            }
        };

        append_runtime_info(hash, node->get_rt_info());

        int port_id = 0;
        // <layers/input>
        if (node->get_input_size() > 0) {
            hash = hash_combine(hash, AttrType::input);
            for (auto& i : node->inputs()) {
                hash = hash_combine(hash, AttrType::port);
                hash = hash_combine(hash_combine(hash, AttrType::id), port_id++);
                hash = hash_combine(hash_combine(hash, AttrType::precision), i.get_element_type().hash());
                hash = hash_combine(hash_combine(hash, AttrType::dimension), i.get_partial_shape().to_string());
                append_runtime_info(hash, i.get_rt_info());
            }
        }
        // <layers/output>
        if ((node->get_output_size() > 0) && !ov::op::util::is_output(node)) {
            hash = hash_combine(hash, AttrType::output);
            for (auto& o : node->outputs()) {
                hash = hash_combine(hash, AttrType::port);
                hash = hash_combine(hash_combine(hash, AttrType::id), port_id++);
                hash = hash_combine(hash_combine(hash, AttrType::precision), o.get_element_type().hash());
                hash = hash_combine(hash_combine(hash, AttrType::dimension), o.get_partial_shape().to_string());
                append_runtime_info(hash, o.get_rt_info());
            }
        }

        // fill <data> general attributes
        {
            SnippetsHasher visitor(hash, node_type_name);
            if (!node->visit_attributes(visitor)) {
                OPENVINO_THROW("Visitor API is not supported in " + node_type_name + " in snippets hash generation");
            }
        }
        rt_info::NodeAuxRTInfoHasher{hash}.serialize(node->get_rt_info());
    }
    // <edges>
    const std::vector<Edge> edge_mapping = create_edge_mapping(layer_ids, ordered_ops);
    hash = hash_combine(hash, AttrType::edges);
    for (const auto& e : edge_mapping) {
        hash = hash_combine(hash, AttrType::edge);
        hash = hash_combine(hash_combine(hash, AttrType::from_layer), e.from_layer);
        hash = hash_combine(hash_combine(hash, AttrType::from_port), e.from_port);
        hash = hash_combine(hash_combine(hash, AttrType::to_layer), e.to_layer);
        hash = hash_combine(hash_combine(hash, AttrType::to_port), e.to_port);
    }

    // Serialize rt info
    hash = hash_combine(hash, AttrType::rt_info);
    for (const auto& it : model.get_rt_info()) {
        hash_rt_info(hash, it.second);
    }
}

}  // namespace

bool Hash::run_on_model(const std::shared_ptr<ov::Model>& m) {
    uint64_t seed = 0;
    std::string name = "net";
    SnippetsHasher visitor(seed, name);
    std::shared_ptr<ov::Model> model(m);  // for complilation error, on_attribute don't accept f
    visitor.on_attribute(name, model);
    m_hash = seed;
    // Return false because we didn't change OpenVINO Model
    return false;
}

Hash::Hash(uint64_t& output_hash_value) : m_hash(output_hash_value) {}

}  // namespace ov::snippets::pass
