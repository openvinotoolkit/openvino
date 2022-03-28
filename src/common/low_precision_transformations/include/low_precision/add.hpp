// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "low_precision/eltwise_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief AddTransformation propagates dequantization subtraction from one input branch to another and
 * propagates dequantization multiplication from the same branch through Add operation.
 *
 * For more details about the transformation, refer to
 * [AddTransformation](@ref openvino_docs_OV_UG_lpt_AddTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API AddTransformation : public EltwiseBaseTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    AddTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
