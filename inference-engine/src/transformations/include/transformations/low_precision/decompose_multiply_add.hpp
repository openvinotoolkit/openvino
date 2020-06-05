// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API DecomposeMultiplyAddTransformation : public LayerTransformation {
public:
    DecomposeMultiplyAddTransformation(const Params& params) : LayerTransformation(params) {}
    ~DecomposeMultiplyAddTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph