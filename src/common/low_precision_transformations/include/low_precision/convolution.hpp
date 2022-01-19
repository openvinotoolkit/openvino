// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "weightable_layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvolutionTransformation propagates dequantization operations through Convolution operation.
 *
 * For more details about the transformation, refer to
 * [ConvolutionTransformation](@ref openvino_docs_IE_DG_lpt_ConvolutionTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ConvolutionTransformation : public WeightableLayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isQuantized(const std::shared_ptr<const Node>& layer) const override;
    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
