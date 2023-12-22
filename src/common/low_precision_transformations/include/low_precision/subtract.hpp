// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief SubtractTransformation propagates dequantization operations through Subtract operation.
 *
 * For more details about the transformation, refer to
 * [SubtractTransformation](@ref openvino_docs_OV_UG_lpt_SubtractTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API SubtractTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("SubtractTransformation", "0");
    SubtractTransformation(const Params& params);
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
