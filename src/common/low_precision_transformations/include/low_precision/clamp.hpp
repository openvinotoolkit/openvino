// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ClampTransformation propagates dequantization operations through Clamp operation.
 *
 * For more details about the transformation, refer to
 * [ClampTransformation](@ref openvino_docs_OV_UG_lpt_ClampTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ClampTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ClampTransformation", "0");
    ClampTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
