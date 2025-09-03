// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/node.hpp"

#include <onnx/onnx_pb.h>

#include "core/attribute.hpp"
#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/tensor.hpp"
#include "input_model.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "translate_session.hpp"

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
    std::shared_ptr<ov::Model> get_subgraph(const std::string name) const;

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

std::shared_ptr<ov::Model> Node::Impl::get_subgraph(const std::string name) const {
    auto it = m_subgraphs.find(name);
    if (it == m_subgraphs.end())
        return nullptr;
    return it->second->decode();
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
std::shared_ptr<ov::Model> Node::Impl::get_attribute_value(const std::string& name,
                                                           std::shared_ptr<ov::Model> default_value) const {
    auto it = std::find_if(std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
        return attribute.get_name() == name;
    });
    if (it == std::end(m_attributes)) {
        return std::forward<std::shared_ptr<ov::Model>>(default_value);
    }
    return get_subgraph(name);
}

template <>
Subgraph Node::Impl::get_attribute_value(const std::string& name) const {
    return get_subgraph_from_attribute(name);
}

template <>
ov::Any Node::get_attribute_value(const std::string& name) const {
    return get_attribute(name).get_any();
}

template <>
std::shared_ptr<ov::Model> Node::Impl::get_attribute_value(const std::string& name) const {
    return get_subgraph(name);
}

ov::OutputVector Node::Impl::get_ov_inputs() const {
    ov::OutputVector result;
    for (const auto& name : m_node_proto->input()) {
        if (!name.empty()) {
            result.push_back(m_graph->get_ov_node_from_cache(name));
        } else {
            result.push_back(std::make_shared<NullNode>()->get_default_output());
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
    : m_pimpl{new Impl{node_proto, graph},
              [](Impl* impl) {
                  delete impl;
              }},
      m_decoder(nullptr),
      m_translate_session(nullptr) {}
Node::Node(const DecoderBaseOperation& decoder, TranslateSession* translate_session)
    : m_pimpl{nullptr,
              [](Impl* impl) {

              }},
      m_decoder(&decoder),
      m_translate_session(translate_session) {}

Node::Node(Node&& other) noexcept
    : m_pimpl{std::move(other.m_pimpl)},
      m_decoder(nullptr),
      m_translate_session(nullptr) {}

Node::Node(const Node& other)
    : m_pimpl{other.m_pimpl != nullptr
                  ? new Impl{other.m_pimpl->node_proto(), other.m_pimpl->graph(), other.get_subgraphs()}
                  : nullptr,
              [](Impl* impl) {
                  delete impl;
              }},
      m_decoder(other.m_decoder),
      m_translate_session(other.m_translate_session) {}

#include <stdexcept>  // For std::runtime_error

ov::OutputVector Node::get_ov_inputs() const {
    if (m_pimpl != nullptr) {
        return m_pimpl->get_ov_inputs();
    } else if (m_decoder != nullptr) {
        ov::OutputVector result;
        auto& known_tensors = m_translate_session->get_tensor_values();
        for (size_t idx = 0; idx < m_decoder->get_input_size(); ++idx) {
            const std::string& name = m_decoder->get_input_tensor_name(idx);
            if (!name.empty()) {
                auto it = known_tensors.find(name);
                FRONT_END_GENERAL_CHECK(it != known_tensors.end());
                result.push_back(it->second);
            } else {
                result.push_back(std::make_shared<NullNode>()->get_default_output());
            }
        }
        return result;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::domain() const {
    static const std::string empty_domain;
    if (m_pimpl != nullptr) {
        return m_pimpl->domain();
    } else if (m_decoder != nullptr) {
        return m_decoder->get_domain();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::op_type() const {
    static const std::string empty_op_type;
    if (m_pimpl != nullptr) {
        return m_pimpl->op_type();
    } else if (m_decoder != nullptr) {
        return m_decoder->get_op_type();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::get_description() const {
    static const std::string empty_description;
    if (m_pimpl != nullptr) {
        return m_pimpl->description();
    } else if (m_decoder != nullptr) {
        // Workaround
        return m_decoder->get_output_tensor_name(0);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::get_name() const {
    static const std::string empty_name;
    if (m_pimpl != nullptr) {
        return m_pimpl->name();
    } else if (m_decoder != nullptr) {
        return m_decoder->get_op_name();
        // Add logic for m_decoder if applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::vector<std::reference_wrapper<const std::string>> Node::get_output_names() const {
    if (m_pimpl != nullptr) {
        return m_pimpl->get_output_names();
    } else if (m_decoder != nullptr) {
        std::vector<std::reference_wrapper<const std::string>> names{};
        names.reserve(m_decoder->get_output_size());
        for (size_t idx = 0; idx < m_decoder->get_output_size(); ++idx) {
            const auto& name = m_decoder->get_output_tensor_name(idx);
            names.push_back(name);
        }
        return {names.begin(), names.end()};
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::input(int index) const {
    static const std::string empty_input;
    if (m_pimpl != nullptr) {
        return m_pimpl->input(index);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_input_tensor_name(index);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

std::size_t Node::get_inputs_size() const {
    if (m_pimpl != nullptr) {
        return m_pimpl->get_inputs_size();
    } else if (m_decoder != nullptr) {
        return m_decoder->get_input_size();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::string& Node::output(int index) const {
    static const std::string empty_output;
    if (m_pimpl != nullptr) {
        return m_pimpl->output(index);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_output_tensor_name(index);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

std::size_t Node::get_outputs_size() const {
    if (m_pimpl != nullptr) {
        return m_pimpl->get_outputs_size();
    } else if (m_decoder != nullptr) {
        return m_decoder->get_output_size();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

bool Node::has_attribute(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->has_attribute(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->has_attribute(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

bool Node::has_subgraphs() const {
    if (m_pimpl != nullptr) {
        return m_pimpl->has_subgraphs();
    } else if (m_decoder != nullptr) {
        // Add logic for m_decoder if applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const std::unordered_map<std::string, std::shared_ptr<Subgraph>>& Node::get_subgraphs() const {
    static const std::unordered_map<std::string, std::shared_ptr<Subgraph>> empty_subgraphs;
    if (m_pimpl != nullptr) {
        return m_pimpl->get_subgraphs();
    } else if (m_decoder != nullptr) {
        // Add logic for m_decoder if applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}
/*
const std::shared_ptr<ov::Model> Node::get_subgraph(const std::string& name) const {

}
*/
std::vector<std::string> Node::get_attribute_names() const {
    if (m_pimpl != nullptr) {
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
    } else if (m_decoder != nullptr) {
        // Add logic for m_decoder if applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

const Attribute& Node::get_attribute(const std::string& name) const {
    if (m_pimpl != nullptr) {
        const auto& node_attributes = m_pimpl->attributes();
        auto found_attr =
            std::find_if(std::begin(node_attributes), std::end(node_attributes), [&name](const Attribute& a) {
                return a.get_name() == name;
            });
        if (found_attr != std::end(node_attributes)) {
            return *found_attr;
        }
    } else if (m_decoder != nullptr) {
        // Add logic for m_decoder if applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

ov::Any Node::get_attribute_any(const std::string& name) const {
    if (m_pimpl != nullptr) {
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
float Node::get_attribute_value(const std::string& name, float default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<float>(name, default_value);
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<float>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
double Node::get_attribute_value(const std::string& name, double default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<double>(name, default_value);
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<double>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::int64_t Node::get_attribute_value(const std::string& name, std::int64_t default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::int64_t>(name, default_value);
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::int64_t>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::string Node::get_attribute_value(const std::string& name, std::string default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::string>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::string>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
Tensor Node::get_attribute_value(const std::string& name, Tensor default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<Tensor>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
SparseTensor Node::get_attribute_value(const std::string& name, SparseTensor default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<SparseTensor>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
Graph Node::get_attribute_value(const std::string& name, Graph default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<Graph>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<float> Node::get_attribute_value(const std::string& name, std::vector<float> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<float>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::vector<float>>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<double> Node::get_attribute_value(const std::string& name, std::vector<double> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<double>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::vector<double>>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::int64_t> Node::get_attribute_value(const std::string& name,
                                                    std::vector<std::int64_t> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::vector<std::int64_t>>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::size_t> Node::get_attribute_value(const std::string& name,
                                                   std::vector<std::size_t> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::vector<std::size_t>>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::string> Node::get_attribute_value(const std::string& name,
                                                   std::vector<std::string> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::string>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (m_decoder->has_attribute(name))
            return m_decoder->get_attribute(name).as<std::vector<std::string>>();
        else
            return default_value;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<Tensor> Node::get_attribute_value(const std::string& name, std::vector<Tensor> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<SparseTensor> Node::get_attribute_value(const std::string& name,
                                                    std::vector<SparseTensor> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<SparseTensor>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<Graph> Node::get_attribute_value(const std::string& name, std::vector<Graph> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<Graph>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::Model> Node::get_attribute_value(const std::string& name,
                                                     std::shared_ptr<ov::Model> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::shared_ptr<ov::Model>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        if (!has_attribute(name)) {
            return default_value;
        }
        auto graph_iterator = m_decoder->get_attribute(name).as<const ov::frontend::onnx::GraphIterator::Ptr>();
        graph_iterator->reset();
        auto input_model =
            std::make_shared<onnx::unify::InputModel>(graph_iterator, m_translate_session->get_input_model().get());
        std::shared_ptr<ov::Model> ov_model(nullptr);
        m_translate_session->translate_graph(input_model, ov_model);
        return ov_model;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

// Repeat the same for the non-default_value overloads:

template <>
float Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<float>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<float>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
double Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<double>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<double>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::int64_t Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::int64_t>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::int64_t>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::size_t Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::size_t>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::size_t>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::string Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::string>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::string>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
Tensor Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<Tensor>(name);
    } else if (m_decoder != nullptr) {
        auto& tensor_decoder = std::dynamic_pointer_cast<ov::frontend::onnx::DecoderBaseTensor>(
            m_decoder->get_attribute(name).as<ov::frontend::onnx::DecoderBase::Ptr>());
        const auto& tensor_meta_info = tensor_decoder->get_tensor_info();
        auto tensor_place = std::make_shared<ov::frontend::onnx::TensorONNXPlace>(
            *m_translate_session->get_input_model().get(),
            tensor_meta_info.m_partial_shape,
            tensor_meta_info.m_element_type,
            std::vector<std::string>{*tensor_meta_info.m_tensor_name},
            tensor_meta_info.m_tensor_data,
            tensor_meta_info.m_tensor_data_size,
            tensor_meta_info.m_external_location);
        return {tensor_place};
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
SparseTensor Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<SparseTensor>(name);
    } else if (m_decoder != nullptr) {
        auto sparse_tensor_info = m_decoder->get_attribute(name).as<ov::frontend::onnx::SparseTensorInfo>();
        FRONT_END_GENERAL_CHECK(sparse_tensor_info.m_indices && sparse_tensor_info.m_values,
                                "Incomplete sparse tensors are not supported");

        auto& values_decoder =
            std::dynamic_pointer_cast<ov::frontend::onnx::DecoderBaseTensor>(sparse_tensor_info.m_values);
        const auto& values_meta_info = values_decoder->get_tensor_info();
        auto values_place = std::make_shared<ov::frontend::onnx::TensorONNXPlace>(
            *m_translate_session->get_input_model().get(),
            values_meta_info.m_partial_shape,
            values_meta_info.m_element_type,
            std::vector<std::string>{*values_meta_info.m_tensor_name},
            values_meta_info.m_tensor_data,
            values_meta_info.m_tensor_data_size,
            values_meta_info.m_external_location);

        auto& indices_decoder =
            std::dynamic_pointer_cast<ov::frontend::onnx::DecoderBaseTensor>(sparse_tensor_info.m_indices);
        const auto& indices_meta_info = indices_decoder->get_tensor_info();
        auto indices_place = std::make_shared<ov::frontend::onnx::TensorONNXPlace>(
            *m_translate_session->get_input_model().get(),
            indices_meta_info.m_partial_shape,
            indices_meta_info.m_element_type,
            std::vector<std::string>{*indices_meta_info.m_tensor_name},
            indices_meta_info.m_tensor_data,
            indices_meta_info.m_tensor_data_size,
            values_meta_info.m_external_location);
        return {values_place, indices_place, sparse_tensor_info.m_partial_shape};
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
Subgraph Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<Subgraph>(name);
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<float> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<float>>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::vector<float>>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<double> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<double>>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::vector<double>>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::int64_t> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::vector<std::int64_t>>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::size_t> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name);
    } else if (m_decoder != nullptr) {
        auto ints = m_decoder->get_attribute(name).as<std::vector<std::int64_t>>();
        return {ints.begin(), ints.end()};
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<std::string> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<std::string>>(name);
    } else if (m_decoder != nullptr) {
        return m_decoder->get_attribute(name).as<std::vector<std::string>>();
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<Tensor> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name);
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<SparseTensor> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<SparseTensor>>(name);
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::vector<Graph> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::vector<Graph>>(name);
    } else if (m_decoder != nullptr) {
        // Non-applicable
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::Model> Node::get_attribute_value(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_value<std::shared_ptr<ov::Model>>(name);
    } else if (m_decoder != nullptr) {
        auto graph_iterator = m_decoder->get_attribute(name).as<const ov::frontend::onnx::GraphIterator::Ptr>();
        graph_iterator->reset();
        auto input_model =
            std::make_shared<onnx::unify::InputModel>(graph_iterator, m_translate_session->get_input_model().get());
        std::shared_ptr<ov::Model> ov_model(nullptr);
        m_translate_session->translate_graph(input_model, ov_model);
        return ov_model;
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

// get_attribute_as_constant specializations

// Calls get_decoder_attribute_as_constant is better to rewrite as ov::Any later
// After GraphIterator will become a default interface
template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::get_decoder_attribute_as_constant(const std::string& name) const {
    const auto value = get_attribute_value<T>(name);
    const ov::element::Type type = ov::element::from<T>();
    return std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_decoder_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name) const {
    const auto values = get_attribute_value<std::vector<int64_t>>(name);
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> Node::get_decoder_attribute_as_constant(const std::string& name,
                                                                              T default_value) const {
    const auto value = get_attribute_value<T>(name, default_value);
    const ov::element::Type type = ov::element::from<T>();
    return std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, value);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<float>(name);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<float>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      float default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<float>(name, default_value);
    } else if (m_decoder != nullptr) {
        float value = default_value;
        if (m_decoder->has_attribute(name)) {
            value = m_decoder->get_attribute(name).as<float>();
        }
        return ov::op::v0::Constant::create(ov::element::f32, {}, {value});
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      float default_value,
                                                                      ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<float>(name, default_value, std::move(type));
    } else if (m_decoder != nullptr) {
        float value = default_value;
        if (m_decoder->has_attribute(name)) {
            value = m_decoder->get_attribute(name).as<float>();
        }
        return ov::op::v0::Constant::create(type == ov::element::dynamic ? ov::element::f32 : type, {}, {value});
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<float>(const std::string& name,
                                                                             ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<float>(name, std::move(type));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<float>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<double>(name);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<double>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      double default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<double>(name, default_value);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<double>(name, default_value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      double default_value,
                                                                      ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<double>(name, default_value, std::move(type));
    } else if (m_decoder != nullptr) {
        double value = default_value;
        if (m_decoder->has_attribute(name)) {
            value = m_decoder->get_attribute(name).as<double>();
        }
        return ov::op::v0::Constant::create(type == ov::element::dynamic ? ov::element::f64 : type, {0}, {value});
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<double>(const std::string& name,
                                                                              ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<double>(name, std::move(type));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<double>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<int64_t>(name);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<int64_t>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      int64_t default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<int64_t>(name, default_value);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<int64_t>(name, default_value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      int64_t default_value,
                                                                      ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<int64_t>(name, default_value, std::move(type));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<int64_t>(name, default_value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<int64_t>(const std::string& name,
                                                                               ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<int64_t>(name, std::move(type));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<int64_t>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name);
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<std::vector<int64_t>>(name);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant<std::vector<int64_t>>(
    const std::string& name,
    ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name, std::move(type));
    } else if (m_decoder != nullptr) {
        auto value = m_decoder->get_attribute(name).as<std::vector<int64_t>>();
        return ov::op::v0::Constant::create(type == ov::element::dynamic ? ov::element::i64 : type,
                                            {value.size()},
                                            value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      std::vector<int64_t> default_value) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name, std::move(default_value));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<std::vector<int64_t>>(name, default_value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

template <>
std::shared_ptr<ov::op::v0::Constant> Node::get_attribute_as_constant(const std::string& name,
                                                                      std::vector<int64_t> default_value,
                                                                      ov::element::Type type) const {
    if (m_pimpl != nullptr) {
        return m_pimpl->template get_attribute_as_constant<std::vector<int64_t>>(name,
                                                                                 std::move(default_value),
                                                                                 std::move(type));
    } else if (m_decoder != nullptr) {
        return get_decoder_attribute_as_constant<std::vector<int64_t>>(name, default_value);
    }
    FRONT_END_NOT_IMPLEMENTED(__FUNCTION__);
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
