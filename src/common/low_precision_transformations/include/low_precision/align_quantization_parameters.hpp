// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API AlignQuantizationParameters;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief AlignQuantizationParameters transformation marks precision preserved operations subgraph by `QuantizationAlignmentAttribute`
 * attribute after FakeQuantize operations.
 *
 * For more details about the transformation, refer to
 * [AlignQuantizationParameters](@ref openvino_docs_OV_UG_lpt_AlignQuantizationParameters) page
 * in the OpenVINO Developer Guide.
 */
class ov::pass::low_precision::AlignQuantizationParameters : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("low_precision::AlignQuantizationParameters");
    AlignQuantizationParameters(const std::vector<ov::element::Type> defaultPrecisions = ov::pass::low_precision::precision_set::get_int8_support());
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
private:
    const std::vector<ov::element::Type> defaultPrecisions;
};
