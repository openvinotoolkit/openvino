// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief NormalizeL2Transformation propagates dequantization operations through NormalizeL2 operation.
 *
 * For more details about the transformation, refer to
 * [NormalizeL2Transformation](@ref openvino_docs_OV_UG_lpt_NormalizeL2Transformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API NormalizeL2Transformation : public LayerTransformation {
public:
    OPENVINO_RTTI("NormalizeL2Transformation", "0", LayerTransformation);
    NormalizeL2Transformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
