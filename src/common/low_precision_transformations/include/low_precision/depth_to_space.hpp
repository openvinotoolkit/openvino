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
 * @brief DepthToSpaceTransformation propagates dequantization operations through DepthToSpace operation.
 *
 * For more details about the transformation, refer to
 * [DepthToSpaceTransformation](@ref openvino_docs_OV_UG_lpt_DepthToSpaceTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API DepthToSpaceTransformation : public TransparentBaseTransformation {
public:
    OPENVINO_RTTI("DepthToSpaceTransformation", "0", TransparentBaseTransformation);
    DepthToSpaceTransformation(const Params& params = Params());
    bool canBeTransformed(const std::shared_ptr<ov::Node>& layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
