// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/variable.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace util
{
    class DictAttributeDeserializer : public ngraph::AttributeVisitor
    {
    public:
        DictAttributeDeserializer(
            const py::dict& attributes,
            std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& variables);

        void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::string>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int8_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int16_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int32_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint8_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint16_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint32_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint64_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<std::string>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int8_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int16_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<float>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<double>>& adapter) override;

        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override;

    protected:
        const py::dict& m_attributes;
        std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& m_variables;
    };

    class DictAttributeSerializer : public ngraph::AttributeVisitor
    {
    public:
        explicit DictAttributeSerializer(const std::shared_ptr<ngraph::Node>& node);

        void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::string>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int8_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int16_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int32_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint8_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint16_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint32_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<uint64_t>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<float>& adapter) override;
        void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<std::string>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int8_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int16_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<float>>& adapter) override;
        void on_adapter(const std::string& name,
                        ngraph::ValueAccessor<std::vector<double>>& adapter) override;

        template <typename T>
        T get_attribute(const std::string& name)
        {
            NGRAPH_CHECK(m_attributes.contains(name),
                         "Couldn't find attribute \"",
                         name,
                         "\" in serialized node attribute dictionary.");
            return m_attributes[name.c_str()].cast<T>();
        }

        py::dict get_attributes() const { return m_attributes; }

    protected:
        py::dict m_attributes;
    };
} // namespace util
