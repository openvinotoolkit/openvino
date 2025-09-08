// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ov {
/**
 * @ingroup ov_transformation_common_api
 * @brief PrecisionPreservedAttribute defines the precision preserved operation. If the attribute is absent, then an operation is
 * not precision preserved.
 *
 * For more details about the attribute, refer to
 * [PrecisionPreservedAttribute](@ref openvino_docs_OV_UG_lpt_PrecisionPreserved) page in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API PrecisionPreservedAttribute : public SharedAttribute<bool> {
public:
    OPENVINO_RTTI("LowPrecision::PrecisionPreserved", "", ov::RuntimeAttribute);

    PrecisionPreservedAttribute() = default;
    PrecisionPreservedAttribute(const bool value);

    std::string to_string() const override;
};

} // namespace ov
