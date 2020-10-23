// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API MultiplyToGroupConvolutionTransformation : public LayerTransformation {
public:
    MultiplyToGroupConvolutionTransformation(const Params& params) : LayerTransformation(params), groupSize(1ul) {}
    ~MultiplyToGroupConvolutionTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool isQuantized(std::shared_ptr<Node> layer) const noexcept override;

    void setGroupSize(const size_t groupSize);
    size_t getGroupSize() const;
private:
    size_t groupSize;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
