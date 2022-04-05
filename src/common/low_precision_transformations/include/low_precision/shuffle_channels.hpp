// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief ShuffleChannelsTransformation propagates dequantization operations through ShuffleChannels operation.
 *
 * For more details about the transformation, refer to
 * [ShuffleChannelsTransformation](@ref openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ShuffleChannelsTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ShuffleChannelsTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
