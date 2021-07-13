// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API SubtractTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    SubtractTransformation(const Params& params);
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
