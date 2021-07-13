// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "weightable_layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ConvolutionTransformation : public WeightableLayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvolutionTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isQuantized(const std::shared_ptr<const Node>& layer) const noexcept override;
    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer) noexcept;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
