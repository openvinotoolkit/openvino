//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace util
{
    class DictAttributeDeserializer : public ngraph::AttributeVisitor
    {
    public:
        DictAttributeDeserializer(const py::dict& attributes);

        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<void>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<bool>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::string>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int8_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int16_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int32_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int64_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint8_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint16_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint32_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint64_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<float>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<double>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<std::string>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int8_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int16_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<float>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<double>>& adapter) override;

    protected:
        const py::dict& m_attributes;
    };

    class DictAttributeSerializer : public ngraph::AttributeVisitor
    {
    public:
        DictAttributeSerializer(const std::shared_ptr<ngraph::Node>& node);

        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<void>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<bool>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::string>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int8_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int16_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int32_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<int64_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint8_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint16_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint32_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<uint64_t>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<float>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<double>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<std::string>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int8_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int16_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override;
        virtual void on_adapter(const std::string& name,
                                ngraph::ValueAccessor<std::vector<float>>& adapter) override;
        virtual void on_adapter(const std::string& name,
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
}
