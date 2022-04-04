// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ngraph {
/**
 * @ingroup ie_transformation_common_api
 * @brief PrecisionsAttribute defines precision which is required for input/output port or an operation.
 *
 * For more details about the attribute, refer to
 * [PrecisionsAttribute](@ref openvino_docs_OV_UG_lpt_Precisions) page in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API PrecisionsAttribute : public SharedAttribute<std::vector<ngraph::element::Type>> {
public:
    OPENVINO_RTTI("LowPrecision::Precisions", "", ov::RuntimeAttribute, 0);
    PrecisionsAttribute(const std::vector<ngraph::element::Type>& precisions);

    static ov::Any create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    // merge attribute instances which can be got from different sources: node, input port or output port
    void merge(std::vector<ov::Any>& attributes);
    // vizualize shared attributes details in VizualizeTree pass
    std::string to_string() const override;
};
} // namespace ngraph
