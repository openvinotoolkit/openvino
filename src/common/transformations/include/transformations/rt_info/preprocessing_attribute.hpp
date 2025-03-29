// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {
TRANSFORMATIONS_API bool is_preprocesing_node(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void set_is_preprocessing_node(std::shared_ptr<Node> node);

/*
 * PreprocessingAttribute attribute indicates that operation can be fused
 * by different fusion transformation for cases when some information is unknown
 * but we can rely on information that operation is a pre-processing operation.
 */
class TRANSFORMATIONS_API PreprocessingAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("preprocessing", "0", ov::RuntimeAttribute);
    PreprocessingAttribute() = default;
    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    };
};
}  // namespace ov
