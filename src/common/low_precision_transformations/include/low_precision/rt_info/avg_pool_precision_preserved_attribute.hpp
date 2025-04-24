// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "openvino/core/node.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

namespace ov {

/**
 * @ingroup ov_transformation_common_api
 * @brief AvgPoolPrecisionPreservedAttribute is utility attribute which is used only during `AvgPool` operation precision
 * preserved property definition.
 *
 * For more details about the attribute, refer to
 * [AvgPoolPrecisionPreservedAttribute](@ref openvino_docs_OV_UG_lpt_AvgPoolPrecisionPreserved) page in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API AvgPoolPrecisionPreservedAttribute : public PrecisionPreservedAttribute {
public:
    OPENVINO_RTTI("LowPrecision::AvgPoolPrecisionPreserved", "", ov::RuntimeAttribute);
    using PrecisionPreservedAttribute::PrecisionPreservedAttribute;
    void merge_attributes(std::vector<ov::Any>& attributes);
    bool is_skipped() const;
    std::string to_string() const override;
};

} // namespace ov
