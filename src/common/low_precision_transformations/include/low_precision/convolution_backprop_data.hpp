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
 * @brief ConvolutionBackpropDataTransformation propagates dequantization operations through ConvolutionBackpropData operation.
 *
 * For more details about the transformation, refer to
 * [ConvolutionBackpropDataTransformation](@ref openvino_docs_OV_UG_lpt_ConvolutionBackpropDataTransformation) page in
 * the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ConvolutionBackpropDataTransformation : public WeightableLayerTransformation {
public:
    ConvolutionBackpropDataTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isQuantized(const std::shared_ptr<const Node>& layer,
        const std::vector<ngraph::element::Type>&defaultPrecisions) const override;
    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer,
        const std::vector<ngraph::element::Type>& defaultPrecisions);

protected:
    size_t getInputChannels(const std::shared_ptr<ngraph::Node> conv) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
