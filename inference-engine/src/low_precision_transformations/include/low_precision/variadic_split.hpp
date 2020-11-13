// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "split.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API VariadicSplitTransformation : public SplitTransformation {
public:
    VariadicSplitTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
protected:
    std::vector<size_t> getConstSplitLengths(
        const OutputVector& inputs,
        const ngraph::Shape& constShape,
        const size_t outputSize) const override;
};
} // namespace low_precision
} // namespace pass
} // namespace ngraph
