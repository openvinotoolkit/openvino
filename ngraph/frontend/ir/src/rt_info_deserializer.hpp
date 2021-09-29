// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ir_frontend/utility.hpp>
#include <istream>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <utils.hpp>

namespace ov {
class RTInfoDeserializer : public ngraph::AttributeVisitor {
public:
    explicit RTInfoDeserializer(const pugi::xml_node& node) : m_node(node) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& value) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        value.set(val);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& value) override {
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

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        adapter.set(stringToType<double>(val));
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        adapter.set(stringToType<int64_t>(val));
    }

    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
        throw ngraph::ngraph_error("Function type is unsupported for rt info deserialization");
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<int32_t> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<int64_t> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<float> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        check_attribute_name(name);
        std::string val;
        if (!getStrAttribute(m_node, name, val))
            return;
        std::vector<std::string> value;
        str_to_container(val, value);
        adapter.set(value);
    }

    void check_attribute_name(const std::string& name) const {
        if (name == "name" || name == "version") {
            throw ngraph::ngraph_error("Attribute key with name: " + name + " is not allowed. Please use another name");
        }
    }

private:
    pugi::xml_node m_node;
};
}  // namespace ov