// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

#include <utils.hpp>
#include <ir_frontend/utility.hpp>
#include <ngraph/ngraph.hpp>

namespace ov {
class RTInfoDeserializer : public ngraph::AttributeVisitor {
public:
    explicit RTInfoDeserializer(const std::string & value) : m_value(value) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& value) override {
        value.set(m_value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& value) override {
        std::string val = m_value;
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
        adapter.set(stringToType<double>(m_value));
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        adapter.set(stringToType<int64_t>(m_value));
    }

    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override {
        IR_THROW("Not implemented");
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        std::vector<int32_t> value;
        str_to_container(m_value, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        std::vector<int64_t> value;
        str_to_container(m_value, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        std::vector<float> value;
        str_to_container(m_value, value);
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> value;
        str_to_container(m_value, value);
        adapter.set(value);
    }
private:
    std::string m_value;
};
}