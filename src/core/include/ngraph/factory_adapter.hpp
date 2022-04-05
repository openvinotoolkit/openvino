// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/factory.hpp"

namespace ngraph {
template <typename BASE_TYPE>

class NGRAPH_DEPRECATED("This class is deprecated and will be removed soon.") FactoryAttributeAdapter
    : public VisitorAdapter {
public:
    FactoryAttributeAdapter(std::shared_ptr<BASE_TYPE>& ref) : m_ref(ref) {}

    /// \brief Hook for extra processing before other attributes
    virtual bool on_start(AttributeVisitor& /* visitor */) {
        return true;
    }
    /// \brief Hook for extra processing after other attributes
    virtual bool on_finish(AttributeVisitor& /* visitor */) {
        return true;
    }
    bool visit_attributes(AttributeVisitor& visitor) override {
        if (on_start(visitor)) {
            std::string type_info_name;
            uint64_t type_info_version;
            if (m_ref) {
                auto& type_info = m_ref->get_type_info();
                type_info_name = type_info.name;
                type_info_version = type_info.version;
            }
            visitor.on_attribute("name", type_info_name);
            visitor.on_attribute("version", type_info_version);
            if (m_ref) {
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
}  // namespace ngraph
