// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include <low_precision/lpt_visibility.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupAvgPoolPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupAvgPoolPrecisionPreserved transformation marks AvgPool operations as precision preserved or not.
 *
 * For more details about the transformation, refer to
 * [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_MarkupAvgPoolPrecisionPreserved) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
