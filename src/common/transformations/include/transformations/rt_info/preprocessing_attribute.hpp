// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <openvino/core/runtime_attribute.hpp>

namespace ov {
NGRAPH_API bool is_preprocesing_node(const std::shared_ptr<ngraph::Node>& node);

NGRAPH_API void set_is_preprocessing_node(std::shared_ptr<ngraph::Node> node);

/*
 * PreprocessingAttribute attribute indicates that operation can be fused
 * by different fusion transformation for cases when some information is unknown
 * but we can rely on information that operation is a pre-processing operation.
 */
class NGRAPH_API PreprocessingAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("preprocessing", "0");
    PreprocessingAttribute() = default;
    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    };
};
}  // namespace ov
