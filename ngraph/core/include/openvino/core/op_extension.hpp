// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/compatibility.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {

class OPENVINO_EXTENSION_API BaseOpExtension : public Extension {
public:
    using Ptr = std::shared_ptr<BaseOpExtension>;
    ~BaseOpExtension() override;
    virtual const ov::DiscreteTypeInfo& type() = 0;
    virtual ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) = 0;
};

template <class T>
class OPENVINO_EXTENSION_API OpExtension : public BaseOpExtension {
    template <typename TYPE, typename std::enable_if<!ngraph::HasTypeInfoMember<TYPE>::value, bool>::type = true>
    ov::DiscreteTypeInfo get_type_info() {
        return TYPE::get_type_info_static();
    }

    template <typename TYPE, typename std::enable_if<ngraph::HasTypeInfoMember<TYPE>::value, bool>::type = true>
    ov::DiscreteTypeInfo get_type_info() {
        NGRAPH_SUPPRESS_DEPRECATED_START
        return TYPE::type_info;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
    const ov::DiscreteTypeInfo ext_type;

public:
    OpExtension() : ext_type(get_type_info<T>()) {
        OPENVINO_ASSERT(ext_type.name != nullptr && ext_type.version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& type() override {
        return ext_type;
    }
    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) override {
        std::shared_ptr<ov::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }
};

}  // namespace ov
