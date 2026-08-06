// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API ConvertLegacyPrecisionAttribute;

}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertLegacyPrecisionAttribute migrates the legacy DisableFP16Compression
 * runtime attribute ("precise") to the new DisablePrecisionConversion attribute.
 * This ensures backward compatibility when loading old IR models that use the
 * legacy attribute.
 */
class ov::pass::ConvertLegacyPrecisionAttribute : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertLegacyPrecisionAttribute");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
