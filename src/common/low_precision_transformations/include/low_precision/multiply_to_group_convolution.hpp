// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/precisions_restriction.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief MultiplyToGroupConvolutionTransformation replace quantized Multiply operations to GroupConvolution to speed up inference.
 *
 * For more details about the transformation, refer to
 * [MultiplyToGroupConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyToGroupConvolutionTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API MultiplyToGroupConvolutionTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("MultiplyToGroupConvolutionTransformation", "0");
    MultiplyToGroupConvolutionTransformation(
        const Params& params = Params(),
        const PrecisionsRestriction::PrecisionsByPort& restrictions = {});
    ~MultiplyToGroupConvolutionTransformation() override {}
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool isQuantized(const std::shared_ptr<const Node>& layer,
        const std::vector<ngraph::element::Type>& defaultPrecisions) const override;
    static bool canBeTransformedToGroupConvolution(const std::shared_ptr<const Node>& layer);
    static bool isDynamicOrScalar(const std::shared_ptr<const Node>& node);

    void setGroupSize(const size_t groupSize);
    size_t getGroupSize() const;
private:
    PrecisionsRestriction::PrecisionsByPort restrictions;
    size_t groupSize;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
