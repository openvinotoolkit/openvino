// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "split.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief VariadicSplitTransformation propagates dequantization operations through VariadicSplit operation.
 *
 * For more details about the transformation, refer to
 * [VariadicSplitTransformation](@ref openvino_docs_OV_UG_lpt_VariadicSplitTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API VariadicSplitTransformation : public SplitTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    VariadicSplitTransformation(const Params& params = Params());
};
} // namespace low_precision
} // namespace pass
} // namespace ngraph
