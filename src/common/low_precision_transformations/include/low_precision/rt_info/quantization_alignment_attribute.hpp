// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "shared_value_attribute.hpp"
#include "attribute_parameters.hpp"

namespace ngraph {
/**
 * @ingroup ie_transformation_common_api
 * @brief QuantizationAlignmentAttribute defines subgraph with the same quantization alignment.
 * FakeQuantize operations are not included. The attribute is used by quantization operations.
 *
 * For more details about the attribute, refer to
 * [QuantizationAlignmentAttribute](@ref openvino_docs_OV_UG_lpt_QuantizationAlignment) page in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API QuantizationAlignmentAttribute : public SharedAttribute<bool> {
public:
    OPENVINO_RTTI("LowPrecision::QuantizationAlignment", "", ov::RuntimeAttribute, 0);
    QuantizationAlignmentAttribute(const bool value = false);

    static ov::Any create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params = AttributeParameters());
    void merge(std::vector<ov::Any>& attributes);
    std::string to_string() const override;
};

} // namespace ngraph
