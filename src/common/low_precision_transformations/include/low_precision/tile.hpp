// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief TileTransformation propagates dequantization operations through Tile operation.
 *
 * For more details about the transformation, refer to
 * [TileTransformation](@ref openvino_docs_IE_DG_lpt_TileTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API TileTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    TileTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
