// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ov {
/**
 * @ingroup ov_transformation_common_api
 * @brief PrecisionsAttribute defines precision which is required for input/output port or an operation.
 *
 * For more details about the attribute, refer to
 * [PrecisionsAttribute](@ref openvino_docs_OV_UG_lpt_Precisions) page in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API PrecisionsAttribute : public SharedAttribute<std::vector<ov::element::Type>> {
public:
    OPENVINO_RTTI("LowPrecision::Precisions", "", ov::RuntimeAttribute);
    PrecisionsAttribute(const std::vector<ov::element::Type>& precisions);

    static ov::Any create(
        const std::shared_ptr<ov::Node>& node,
        const AttributeParameters& params);
    // merge attribute instances which can be got from different sources: node, input port or output port
    void merge_attributes(std::vector<ov::Any>& attributes);
    // vizualize shared attributes details in VizualizeTree pass
    std::string to_string() const override;
};
} // namespace ov
