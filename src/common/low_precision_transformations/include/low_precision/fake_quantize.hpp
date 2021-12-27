// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"
#include "low_precision/fuse_fake_quantize.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API FakeQuantizeTransformation : public LayerTransformation {
public:
    NGRAPH_RTTI_DECLARATION;
    FakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool checkElementwise(const std::shared_ptr<Node>& eltwise);

private:
    std::shared_ptr<opset1::FakeQuantize> fuseElementwise(
            TransformationContext& context,
            MatcherPass* matcherPass,
            const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
