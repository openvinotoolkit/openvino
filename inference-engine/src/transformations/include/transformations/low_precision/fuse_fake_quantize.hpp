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

class TRANSFORMATIONS_API FuseFakeQuantizeTransformation : public LayerTransformation {
public:
    FuseFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {}
    ~FuseFakeQuantizeTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

private:
    std::shared_ptr<opset1::FakeQuantize> handle(
        TransformationContext& context,
        const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const;

    static std::shared_ptr<Node> updateShape(std::shared_ptr<Node> op, const Shape& targetShape);
    static std::shared_ptr<Node> getData(const std::shared_ptr<Node>& eltwise);
    static std::shared_ptr<opset1::Constant> getConstant(const std::shared_ptr<Node>& eltwise);
    static bool eltwiseWithConstant(const std::shared_ptr<Node>& eltwise);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
