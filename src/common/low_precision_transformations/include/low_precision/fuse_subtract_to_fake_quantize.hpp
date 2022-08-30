// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief FuseSubtractToFakeQuantizeTransformation fuses Subtract operation to FakeQuantize.
 *
 * For more details about the transformation, refer to
 * [FuseSubtractToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseSubtractToFakeQuantizeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FuseSubtractToFakeQuantizeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("FuseSubtractToFakeQuantizeTransformation", "0");
    FuseSubtractToFakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
