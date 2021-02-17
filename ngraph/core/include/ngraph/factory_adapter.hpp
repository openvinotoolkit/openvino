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

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/factory.hpp"

namespace ngraph
{
    template <typename BASE_TYPE>
    class FactoryAttributeAdapter : public VisitorAdapter
    {
    public:
        FactoryAttributeAdapter(std::shared_ptr<BASE_TYPE>& ref)
            : m_ref(ref)
        {
        }

        /// \brief Hook for extra processing before other attributes
        virtual bool on_start(AttributeVisitor& /* visitor */) { return true; }
        /// \brief Hook for extra processing after other attributes
        virtual bool on_finish(AttributeVisitor& /* visitor */) { return true; }
        bool visit_attributes(AttributeVisitor& visitor) override
        {
            if (on_start(visitor))
            {
                std::string type_info_name;
                uint64_t type_info_version;
                if (m_ref)
                {
                    auto& type_info = m_ref->get_type_info();
                    type_info_name = type_info.name;
                    type_info_version = type_info.version;
                }
                visitor.on_attribute("name", type_info_name);
                visitor.on_attribute("version", type_info_version);
                if (m_ref)
                {
                    visitor.start_structure("value");
                    m_ref->visit_attributes(visitor);
                    visitor.finish_structure();
                }
                on_finish(visitor);
            }
            return true;
        }

    protected:
        std::shared_ptr<BASE_TYPE>& m_ref;
    };
}
