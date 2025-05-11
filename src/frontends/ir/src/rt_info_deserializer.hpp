// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <istream>
#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "utils.hpp"

namespace ov {
class RTInfoDeserializer : public ov::AttributeVisitor {
public:
    explicit RTInfoDeserializer(const pugi::xml_node& node) : m_node(node) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& value) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        value.set(val);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& value) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::transform(val.begin(), val.end(), val.begin(), [](char ch) {
            return std::tolower(static_cast<unsigned char>(ch));
        });
        std::set<std::string> true_names{"true", "1"};
        std::set<std::string> false_names{"false", "0"};

        bool is_true = true_names.find(val) != true_names.end();
        bool is_false = false_names.find(val) != false_names.end();

        if (!is_true && !is_false)
            return;
        value.set(is_true);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        adapter.set(stringToType<double>(val));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        adapter.set(stringToType<int64_t>(val));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW("Model type is unsupported for rt info deserialization");
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<int32_t> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<int64_t> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<float> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<uint64_t> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<std::string> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void check_attribute_name(const std::string& name) const {
        OPENVINO_ASSERT(name != "name" && name != "version",
                        "Attribute key with name: ",
                        name,
                        " is not allowed. Please use another name");
    }

private:
    pugi::xml_node m_node;
};
}  // namespace ov
