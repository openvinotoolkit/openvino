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
 * @brief FuseMultiplyToFakeQuantizeTransformation fuses Multiply operation to FakeQuantize.
 *
 * For more details about the transformation, refer to
 * [FuseMultiplyToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseMultiplyToFakeQuantizeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FuseMultiplyToFakeQuantizeTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    FuseMultiplyToFakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
