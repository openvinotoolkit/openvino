// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

using namespace std;

ov::AttributeVisitor::~AttributeVisitor() = default;

void ov::AttributeVisitor::start_structure(const string& name) {
    m_context.push_back(name);
}

string ov::AttributeVisitor::finish_structure() {
    string result = m_context.back();
    m_context.pop_back();
    return result;
}

string ov::AttributeVisitor::get_name_with_context() {
    ostringstream result;
    string sep = "";
    for (const auto& c : m_context) {
        result << sep << c;
        sep = ".";
    }
    return result.str();
}

void ov::AttributeVisitor::on_adapter(const std::string& name, VisitorAdapter& adapter) {
    adapter.visit_attributes(*this);
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<void*>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<string>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<bool>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<int8_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<int16_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<int32_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<int64_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint8_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint16_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint32_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<uint64_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<float>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<double>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int8_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int16_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int32_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<int64_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint8_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint16_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint32_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<uint64_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<float>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<double>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::vector<string>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

constexpr const char* ov::AttributeVisitor::invalid_node_id;

void ov::AttributeVisitor::register_node(const std::shared_ptr<ov::Node>& node, node_id_t id) {
    if (id == invalid_node_id) {
        id = node->get_friendly_name();
    }
    m_id_node_map[id] = node;
    m_node_id_map[node] = std::move(id);
}

std::shared_ptr<ov::Node> ov::AttributeVisitor::get_registered_node(node_id_t id) {
    auto it = m_id_node_map.find(id);
    return it == m_id_node_map.end() ? shared_ptr<ov::Node>() : it->second;
}

ov::AttributeVisitor::node_id_t ov::AttributeVisitor::get_registered_node_id(const std::shared_ptr<ov::Node>& node) {
    auto it = m_node_id_map.find(node);
    return it == m_node_id_map.end() ? invalid_node_id : it->second;
}
