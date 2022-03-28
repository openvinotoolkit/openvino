// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "layer_transformation.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitTransformation propagates dequantization operations through Split operation.
 *
 * For more details about the transformation, refer to
 * [SplitTransformation](@ref openvino_docs_OV_UG_lpt_SplitTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API SplitTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    SplitTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    void updateOutputs(
        TransformationContext& context,
        std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
        std::shared_ptr<ngraph::Node> originalNode) const;
};
} // namespace low_precision
} // namespace pass
} // namespace ngraph
