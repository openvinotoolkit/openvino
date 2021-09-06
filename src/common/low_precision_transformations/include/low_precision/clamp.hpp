// Copyright (C) 2018-2021 Intel Corporation
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
 * @brief ClampTransformation propagates dequantization operations through Clamp operation.
 */
class LP_TRANSFORMATIONS_API ClampTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ClampTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
