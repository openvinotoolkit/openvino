// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transparent_base_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief BroadcastTransformation propagates dequantization operations through Broadcast operation.
 *
 * For more details about the transformation, refer to
 * [BroadcastTransformation](@ref openvino_docs_OV_UG_lpt_BroadcastTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API BroadcastTransformation : public TransparentBaseTransformation {
public:
    OPENVINO_RTTI("BroadcastTransformation", "0", TransparentBaseTransformation);
    BroadcastTransformation(const Params& params = Params());
    bool canBeTransformed(const std::shared_ptr<ov::Node>& layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
