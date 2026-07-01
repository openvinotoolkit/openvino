// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>

#include "lpt_visibility.hpp"
#include "openvino/pass/pass.hpp"
#include "quantization_details.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FQStrippingTransformation strips FakeQuantize operations with specified levels
 * by replacing them with Clamp operations.
 */
class LP_TRANSFORMATIONS_API FQStrippingTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("low_precision::FQStrippingTransformation");
    FQStrippingTransformation(const std::set<size_t>& levels_to_strip, bool need_weights_adjustment = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const std::set<size_t> levels_to_strip;
    const bool need_weights_adjustment;
};

} // namespace low_precision
} // namespace pass
} // namespace ov