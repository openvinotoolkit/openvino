// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief UnsqueezeTransformation propagates dequantization operations through Unsqueeze operation.
 *
 * For more details about the transformation, refer to
 * [UnsqueezeTransformation](@ref openvino_docs_OV_UG_lpt_UnsqueezeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API UnsqueezeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("UnsqueezeTransformation", "0");
    UnsqueezeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
