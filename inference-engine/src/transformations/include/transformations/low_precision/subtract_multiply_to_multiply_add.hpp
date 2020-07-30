// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "transformations/low_precision/layer_transformation.hpp"
#include "transformations/low_precision/eltwise_base_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API SubtrcatMultiplyToMultiplyAddTransformation : public EltwiseBaseTransformation {
public:
    SubtrcatMultiplyToMultiplyAddTransformation(const Params& params) : EltwiseBaseTransformation(params) {}
    ~SubtrcatMultiplyToMultiplyAddTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
