// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transparent_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief DepthToSpaceTransformation propagates dequantization operations through DepthToSpace operation.
 *
 * For more details about the transformation, refer to
 * [DepthToSpaceTransformation](@ref openvino_docs_OV_UG_lpt_DepthToSpaceTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API DepthToSpaceTransformation : public TransparentBaseTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    DepthToSpaceTransformation(const Params& params = Params());
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
