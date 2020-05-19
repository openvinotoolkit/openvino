//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <utility>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    template <typename T>
    class ValueAccessor;

    /// \brief Visits the attributes of a node.
    ///
    /// Attributes are the values set when building a graph which are not
    /// computed as the graph executes. Values computed from the graph topology and attributes
    /// during compilation are not attributes.
    class NGRAPH_API AttributeVisitor
    {
    public:
        virtual ~AttributeVisitor() {}
        // Must implement these methods
        virtual void on_attribute(const std::string& name, std::string& value) = 0;
        virtual void on_attribute(const std::string& name, bool& value) = 0;
        virtual void on_attribute(const std::string& name, void* data, size_t size) {}
        virtual void on_adapter(const std::string& name, ValueAccessor<void>& adapter) = 0;
        // The remaining adapter methods fall back on the void adapter if not implemented
        virtual void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        };
        virtual void on_adapter(const std::string& name, ValueAccessor<int8_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<int16_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<int32_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<uint8_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<uint16_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<uint32_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<uint64_t>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<float>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<double>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int8_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int16_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int32_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<int64_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint8_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint16_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint32_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<uint64_t>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<double>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        virtual void on_adapter(const std::string& name,
                                ValueAccessor<std::vector<std::string>>& adapter)
        {
            on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
        }
        // Use an adapter for non-primitive types
        template <typename T>
        // typename std::enable_if<std::is_class<T>::value, void>::type
        void on_attribute(const std::string& name, T& value)
        {
            AttributeAdapter<T> adapter(value);
            on_adapter(name, adapter);
        }
        void on_attribute(const std::string& name, op::AutoBroadcastSpec& value)
        {
            AttributeAdapter<op::AutoBroadcastType> adapter(value.m_type);
            on_adapter(name, adapter);
        }
        void on_attribute(const std::string& name, op::BroadcastModeSpec& value)
        {
            AttributeAdapter<op::BroadcastType> adapter(value.m_type);
            on_adapter(name, adapter);
        }
    };
}
