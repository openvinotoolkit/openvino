// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief BatchToSpaceTransformation propagates dequantization operations through BatchToSpace operation.
 *
 * For more details about the transformation, refer to
 * [BatchToSpaceTransformation](@ref openvino_docs_OV_UG_lpt_BatchToSpaceTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API BatchToSpaceTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("BatchToSpaceTransformation", "0");
    BatchToSpaceTransformation(const Params& params = Params());
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
