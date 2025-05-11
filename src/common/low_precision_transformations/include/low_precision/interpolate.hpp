// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transparent_base_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief InterpolateTransformation propagates dequantization operations through Interpolate operation.
 *
 * For more details about the transformation, refer to
 * [InterpolateTransformation](@ref openvino_docs_OV_UG_lpt_InterpolateTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API InterpolateTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("InterpolateTransformation", "0", LayerTransformation);
    InterpolateTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
