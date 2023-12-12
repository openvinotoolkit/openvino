// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

class TRANSFORMATIONS_API KeepOriginalPrecision;

TRANSFORMATIONS_API void set_keep_original_precision_attribute(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void remove_keep_original_precision_attribute(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool has_keep_original_precision_attribute(const std::shared_ptr<const Node>& node);

TRANSFORMATIONS_API const KeepOriginalPrecision& get_keep_original_precision_attribute(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief KeepOriginalPrecision class represents runtime info attribute that marks a node
 * that its input and output precision cannot be changed by ConvertPrecision
 */
class KeepOriginalPrecision : public RuntimeAttribute {
public:
    OPENVINO_RTTI("keep_original_precision", "0");

    KeepOriginalPrecision(const std::shared_ptr<const Node>& node);

    bool is_copyable() const override {
        return true;
    }

    const element::TypeVector& get_input_types() const {
        return m_input_types;
    }

    const element::TypeVector& get_output_types() const {
        return m_output_types;
    }

private:
    element::TypeVector m_input_types;
    element::TypeVector m_output_types;
};

}  // namespace ov
