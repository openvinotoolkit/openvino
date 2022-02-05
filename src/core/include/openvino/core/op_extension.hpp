// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {

/**
 * @brief The base interface for OpenVINO operation extensions
 */
class OPENVINO_API BaseOpExtension : public Extension {
public:
    using Ptr = std::shared_ptr<BaseOpExtension>;
    /**
     * @brief Returns the type info of operation
     *
     * @return ov::DiscreteTypeInfo
     */
    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    /**
     * @brief Method creates an OpenVINO operation
     *
     * @param inputs vector of input ports
     * @param visitor attribute visitor which allows to read necessaty arguments
     *
     * @return vector of output ports
     */
    virtual ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const = 0;

    /**
     * @brief Destructor
     */
    ~BaseOpExtension() override;
};

/**
 * @brief The default implementation of OpenVINO operation extensions
 */
template <class T>
class OpExtension : public BaseOpExtension {
public:
    /**
     * @brief Default constructor
     */
    OpExtension() {
        const auto& ext_type = get_type_info();
        OPENVINO_ASSERT(ext_type.name != nullptr && ext_type.version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return T::get_type_info_static();
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        std::shared_ptr<ov::Node> node = std::make_shared<T>();

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }
};

}  // namespace ov
