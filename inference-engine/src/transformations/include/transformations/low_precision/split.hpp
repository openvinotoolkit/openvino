// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "layer_transformation.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API SplitTransformation : public LayerTransformation {
public:
    SplitTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher& m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    void updateOutputs(
        TransformationContext& context,
        std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
        std::shared_ptr<ngraph::Node> originalNode) const;
protected:
    ngraph::Shape getConstSplitShape(
        const std::vector<size_t>& constSplitLengths,
        const ngraph::Shape& constShape, const size_t axis,
        const size_t idx) const;
    virtual std::vector<size_t> getConstSplitLengths(
        const OutputVector& inputs,
        const ngraph::Shape& constShape,
        const size_t outputSize) const;
};
} // namespace low_precision
} // namespace pass
} // namespace ngraph
