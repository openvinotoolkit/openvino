// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "low_precision/eltwise_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API AddTransformation : public EltwiseBaseTransformation {
public:
    AddTransformation(const Params& params) : EltwiseBaseTransformation(params) {}
    ~AddTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
