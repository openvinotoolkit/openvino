// Copyright (C) 2018-2022 Intel Corporation
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
 * @brief FakeQuantizeDecompositionTransformation decomposes FakeQuantize operations to quantize
 * (FakeQuantize with changes output intervals and low precision output type) and dequantize operations.
 *
 * For more details about the transformation, refer to
 * [FakeQuantizeDecompositionTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FakeQuantizeDecompositionTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    FakeQuantizeDecompositionTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
