// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/pass/pass.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API AlignQuantizationParameters;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief AlignQuantizationParameters transformation marks precision preserved operations subgraph by `QuantizationAlignmentAttribute`
 * attribute after FakeQuantize operations.
 *
 * For more details about the transformation, refer to
 * [AlignQuantizationParameters](@ref openvino_docs_OV_UG_lpt_AlignQuantizationParameters) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::AlignQuantizationParameters : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AlignQuantizationParameters(const std::vector<ngraph::element::Type> defaultPrecisions = ngraph::pass::low_precision::precision_set::int8_support);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    const std::vector<ngraph::element::Type> defaultPrecisions;
};
