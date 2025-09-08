// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/node.hpp"

#include <onnx/onnx_pb.h>

#include "core/attribute.hpp"
#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/tensor.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class Node::Impl {
public:
    Impl() = delete;

    Impl(const NodeProto& node_proto, Graph* graph)
        : m_node_proto{&node_proto},
          m_name{node_proto.has_name() ? node_proto.name() : ""},
          m_domain{get_node_domain(node_proto)},
          m_graph{graph},
          m_output_names{std::begin(node_proto.output()), std::end(node_proto.output())} {
        const auto& attributes = node_proto.attribute();
        m_attributes.reserve(attributes.size());
        for (const auto& attr_proto : attributes) {
            m_attributes.emplace_back(attr_proto, m_graph->model_dir(), m_graph->get_mmap_cache());
            const auto& attribute = m_attributes.back();
            if (attribute.is_graph())
                m_subgraphs.insert({attribute.get_name(), std::make_shared<Subgraph>(attribute.get_subgraph(m_graph))});
        }
    }

    Impl(const NodeProto& node_proto,
         Graph* graph,
         const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& subgraphs)
        : m_node_proto{&node_proto},
          m_name{node_proto.has_name() ? node_proto.name() : ""},
          m_domain{get_node_domain(node_proto)},
          m_graph{graph},
          m_output_names{std::begin(node_proto.output()), std::end(node_proto.output())},
          m_subgraphs(subgraphs) {
        for (const auto& attr_proto : node_proto.attribute()) {
            m_attributes.emplace_back(attr_proto, m_graph->model_dir(), m_graph->get_mmap_cache());
        }
    }

    const std::vector<Attribute>& attributes() const;
    ov::OutputVector get_ov_inputs() const;

    const std::string& domain() const;
    const std::string& op_type() const;
    const std::string& name() const;

    const std::string& description() const;
    const std::vector<std::reference_wrapper<const std::string>>& get_output_names() const;
    const std::string& input(int index) const;
    std::size_t get_inputs_size() const;
    const std::string& output(int index) const;
    std::size_t get_outputs_size() const;

    bool has_attribute(const std::string& name) const;

    bool has_subgraphs() const;
    const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& get_subgraphs() const;

    template <typename T>
    T get_attribute_value(const std::string& name, T default_value) const;

    template <typename T>
    T get_attribute_value(const std::string& name) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name,
                                                                    ov::element::Type type) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name, T default_value) const;

    template <typename T>
    std::shared_ptr<ov::op::v0::Constant> get_attribute_as_constant(const std::string& name,
                                                                    T default_value,
                                                                    ov::element::Type type) const;

    const NodeProto& node_proto() const;
    Graph* graph() const;

private:
    Subgraph get_subgraph_from_attribute(const std::string& name) const;

    const NodeProto* m_node_proto;
    std::string m_name;
    std::string m_domain;
    Graph* m_graph;
    std::vector<Attribute> m_attributes;
    std::vector<std::reference_wrapper<const std::string>> m_output_names;
    mutable std::string m_description;

    std::unordered_map<std::string, std::shared_ptr<Subgraph>> m_subgraphs;
};

const NodeProto& Node::Impl::node_proto() const {
    return *m_node_proto;
}
Graph* Node::Impl::graph() const {
    return m_graph;
}
const std::vector<Attribute>& Node::Impl::attributes() const {
    return m_attributes;
}
const std::string& Node::Impl::domain() const {
    return m_domain;
}
const std::string& Node::Impl::op_type() const {
    return m_node_proto->op_type();
}
const std::string& Node::Impl::name() const {
    return m_name;
}
const std::vector<std::reference_wrapper<const std::string>>& Node::Impl::get_output_names() const {
    return m_output_names;
}

const std::string& Node::Impl::input(int index) const {
    return m_node_proto->input(index);
}

std::size_t Node::Impl::get_inputs_size() const {
    return m_node_proto->input_size();
}

const std::string& Node::Impl::output(int index) const {
    return m_node_proto->output(index);
}

std::size_t Node::Impl::get_outputs_size() const {
    return m_output_names.size();
}

bool Node::Impl::has_attribute(const std::string& name) const {
    auto it = std::find_if(std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
        return attribute.get_name() == name;
    });
    return it != std::end(m_attributes);
}

Subgraph Node::Impl::get_subgraph_from_attribute(const std::string& name) const {
    auto it = std::find_if(std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
        return attribute.get_name() == name;
    });
    if (it == std::end(m_attributes)) {
        throw error::node::UnknownAttribute{this->name(), name};
    }
    return it->get_subgraph(m_graph);
}

bool Node::Impl::has_subgraphs() const {
    return m_subgraphs.size() > 0;
}

const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& Node::Impl::get_subgraphs() const {
    return m_subgraphs;
}

template <typename T>
T Node::Impl::get_attribute_value(const std::string& name, T default_value) const {
    auto it = std::find_if(std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
        return attribute.get_name() == name;
    });
    if (it == std::end(m_attributes)) {
        return std::forward<T>(default_value);
    }
    return it->template get_value<T>();
}

template <typename T>
T Node::Impl::get_attribute_value(const std::string& name) const {
    auto it = std::find_if(std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
        return attribute.get_name() == name;
    });
    if (it == std::end(m_attributes)) {
        throw error::node::UnknownAttribute{this->name(), name};
    }
    return it->template get_value<T>();
}

template <>
Subgraph Node::Impl::get_attribute_value(const std::string& name) const {
    return get_subgraph_from_attribute(name);
}

template <>
ov::Any Node::get_attribute_value(const std::string& name) const {
    return get_attribute(name).get_any();
}

ov::OutputVector Node::Impl::get_ov_inputs() const {
    ov::OutputVector result;
    for (const auto& name : m_node_proto->input()) {
        if (!name.empty()) {
            result.push_back(m_graph->get_ov_node_from_cache(name));
        } else {
            result.push_back(std::make_shared<NullNode>()->output(0));
        }
    }
    return result;
}

const std::string& Node::Impl::description() const {
    if (m_description.empty()) {
        if (!name().empty()) {
            m_description = name();
        } else {
            for (std::size_t index = 0; index < m_output_names.size(); ++index) {
                m_description += (index != 0 ? ", " : "") + m_output_names.at(index).get();
            }
        }
    }
    return m_description;
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name) const {
    const auto value = get_attribute_value<T>(name);
    const ov::element::Type type = ov::element::from<T>();
    return std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, value);
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name,
                                                                            T default_value) const {
    const auto value = get_attribute_value<T>(name, default_value);
    const ov::element::Type type = ov::element::from<T>();
    return std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, value);
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name,
                                                                            T default_value,
                                                                            ov::element::Type type) const {
    const auto value = get_attribute_value<T>(name, default_value);
    return std::make_shared<ov::op::v0::Constant>(type == ov::element::dynamic ? ov::element::from<T>() : type,
                                                  ov::Shape{},
                                                  value);
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name,
                                                                            ov::element::Type type) const {
    const auto value = get_attribute_value<T>(name);
    return std::make_shared<ov::op::v0::Constant>(type == ov::element::dynamic ? ov::element::from<T>() : type,
                                                  ov::Shape{},
                                                  value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name) const {
    const auto value = get_attribute_value<std::vector<int64_t>>(name);
    return ov::op::v0::Constant::create(ov::element::i64, {value.size()}, value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name,
    ov::element::Type type) const {
    const auto value = get_attribute_value<std::vector<int64_t>>(name);
    return ov::op::v0::Constant::create(type == ov::element::dynamic ? ov::element::i64 : type, {value.size()}, value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name,
                                                                            std::vector<int64_t> default_value) const {
    const auto value = get_attribute_value<std::vector<int64_t>>(name, default_value);
    return ov::op::v0::Constant::create(ov::element::i64, {value.size()}, value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::Impl::get_attribute_as_constant(const std::string& name,
                                                                            std::vector<int64_t> default_value,
                                                                            ov::element::Type type) const {
    const auto value = get_attribute_value<std::vector<int64_t>>(name, default_value);
    return ov::op::v0::Constant::create(type != ov::element::dynamic ? type : ov::element::i64, {value.size()}, value);
}

Node::Node(const NodeProto& node_proto, Graph* graph)
    : m_pimpl{new Impl{node_proto, graph}, [](Impl* impl) {
                  delete impl;
              }} {}

Node::Node(Node&& other) noexcept : m_pimpl{std::move(other.m_pimpl)} {}

Node::Node(const Node& other)
    : m_pimpl{new Impl{other.m_pimpl->node_proto(), other.m_pimpl->graph(), other.get_subgraphs()}, [](Impl* impl) {
                  delete impl;
              }} {}

ov::OutputVector Node::get_ov_inputs() const {
    return m_pimpl->get_ov_inputs();
}
const std::string& Node::domain() const {
    return m_pimpl->domain();
}
const std::string& Node::op_type() const {
    return m_pimpl->op_type();
}
const std::string& Node::get_description() const {
    return m_pimpl->description();
}
const std::string& Node::get_name() const {
    return m_pimpl->name();
}
const std::vector<std::reference_wrapper<const std::string>>& Node::get_output_names() const {
    return m_pimpl->get_output_names();
}

const std::string& Node::input(int index) const {
    return m_pimpl->input(index);
}

std::size_t Node::get_inputs_size() const {
    return m_pimpl->get_inputs_size();
}

const std::string& Node::output(int index) const {
    return m_pimpl->output(index);
}

std::size_t Node::get_outputs_size() const {
    return m_pimpl->get_outputs_size();
}

bool Node::has_attribute(const std::string& name) const {
    return m_pimpl->has_attribute(name);
}

bool Node::has_subgraphs() const {
    return m_pimpl->has_subgraphs();
}

const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& Node::get_subgraphs() const {
    return m_pimpl->get_subgraphs();
}

std::vector<std::string> Node::get_attribute_names() const {
    std::vector<std::string> attr_names;
    const auto& node_attributes = m_pimpl->attributes();
    attr_names.reserve(node_attributes.size());
    std::transform(std::begin(node_attributes),
                   std::end(node_attributes),
                   std::back_inserter(attr_names),
                   [](const Attribute& a) {
                       return a.get_name();
                   });
    return attr_names;
}

const Attribute& Node::get_attribute(const std::string& name) const {
    const auto& node_attributes = m_pimpl->attributes();
    auto found_attr = std::find_if(std::begin(node_attributes), std::end(node_attributes), [&name](const Attribute& a) {
        return a.get_name() == name;
    });
    if (found_attr == std::end(node_attributes)) {
        throw error::node::UnknownAttribute{this->get_name(), name};
    }
    return *found_attr;
}

template <>
float Node::get_attribute_value(const std::string& name, float default_value) const {
    return m_pimpl->template get_attribute_value<float>(name, default_value);
}

template <>
double Node::get_attribute_value(const std::string& name, double default_value) const {
    return m_pimpl->template get_attribute_value<double>(name, default_value);
}

template <>
std::int64_t Node::get_attribute_value(const std::string& name, std::int64_t default_value) const {
    return m_pimpl->template get_attribute_value<std::int64_t>(name, default_value);
}

template <>
std::string Node::get_attribute_value(const std::string& name, std::string default_value) const {
    return m_pimpl->template get_attribute_value<std::string>(name, std::move(default_value));
}

template <>
Tensor Node::get_attribute_value(const std::string& name, Tensor default_value) const {
    return m_pimpl->template get_attribute_value<Tensor>(name, std::move(default_value));
}

template <>
SparseTensor Node::get_attribute_value(const std::string& name, SparseTensor default_value) const {
    return m_pimpl->template get_attribute_value<SparseTensor>(name, std::move(default_value));
}

template <>
Graph Node::get_attribute_value(const std::string& name, Graph default_value) const {
    return m_pimpl->template get_attribute_value<Graph>(name, std::move(default_value));
}

template <>
std::vector<float> Node::get_attribute_value(const std::string& name, std::vector<float> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<float>>(name, std::move(default_value));
}

template <>
std::vector<double> Node::get_attribute_value(const std::string& name, std::vector<double> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<double>>(name, std::move(default_value));
}

template <>
std::vector<std::int64_t> Node::get_attribute_value(const std::string& name,
                                                    std::vector<std::int64_t> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name, std::move(default_value));
}

template <>
std::vector<std::size_t> Node::get_attribute_value(const std::string& name,
                                                   std::vector<std::size_t> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name, std::move(default_value));
}

template <>
std::vector<std::string> Node::get_attribute_value(const std::string& name,
                                                   std::vector<std::string> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<std::string>>(name, std::move(default_value));
}

template <>
std::vector<Tensor> Node::get_attribute_value(const std::string& name, std::vector<Tensor> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name, std::move(default_value));
}

template <>
std::vector<SparseTensor> Node::get_attribute_value(const std::string& name,
                                                    std::vector<SparseTensor> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<SparseTensor>>(name, std::move(default_value));
}

template <>
std::vector<Graph> Node::get_attribute_value(const std::string& name, std::vector<Graph> default_value) const {
    return m_pimpl->template get_attribute_value<std::vector<Graph>>(name, std::move(default_value));
}

template <>
float Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<float>(name);
}

template <>
double Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<double>(name);
}

template <>
std::int64_t Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::int64_t>(name);
}

template <>
std::size_t Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::size_t>(name);
}

template <>
std::string Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::string>(name);
}

template <>
Tensor Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<Tensor>(name);
}

template <>
SparseTensor Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<SparseTensor>(name);
}

template <>
Subgraph Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<Subgraph>(name);
}

template <>
std::vector<float> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<float>>(name);
}

template <>
std::vector<double> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<double>>(name);
}

template <>
std::vector<std::int64_t> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name);
}

template <>
std::vector<std::size_t> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name);
}

template <>
std::vector<std::string> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<std::string>>(name);
}

template <>
std::vector<Tensor> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name);
}

template <>
std::vector<SparseTensor> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<SparseTensor>>(name);
}

template <>
std::vector<Graph> Node::get_attribute_value(const std::string& name) const {
    return m_pimpl->template get_attribute_value<std::vector<Graph>>(name);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(const std::string& name) const {
    return m_pimpl->template get_attribute_as_constant<float>(name);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      float default_value) const {
    return m_pimpl->template get_attribute_as_constant<float>(name, default_value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      float default_value,
                                                                      ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<float>(name, default_value, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(const std::string& name,
                                                                             ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<float>(name, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(const std::string& name) const {
    return m_pimpl->template get_attribute_as_constant<double>(name);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      double default_value) const {
    return m_pimpl->template get_attribute_as_constant<double>(name, default_value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      double default_value,
                                                                      ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<double>(name, default_value, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(const std::string& name,
                                                                              ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<double>(name, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(const std::string& name) const {
    return m_pimpl->template get_attribute_as_constant<int64_t>(name);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      int64_t default_value) const {
    return m_pimpl->template get_attribute_as_constant<int64_t>(name, default_value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      int64_t default_value,
                                                                      ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<int64_t>(name, default_value, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(const std::string& name,
                                                                               ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<int64_t>(name, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name) const {
    return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name,
    ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name, std::move(type));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      std::vector<int64_t> default_value) const {
    return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name, std::move(default_value));
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      std::vector<int64_t> default_value,
                                                                      ov::element::Type type) const {
    return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name,
                                                                             std::move(default_value),
                                                                             std::move(type));
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
