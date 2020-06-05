// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {


class TRANSFORMATIONS_API FakeQuantizeTransformation : public LayerTransformation {
public:
    FakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {}
    ~FakeQuantizeTransformation() override {};
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    void setWeightsToConst(const bool weightsToConst);
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

protected:
    // TODO: isolate to a dedicated transformation?
    // void fuseScaleShift(TransformationContext& context, CNNLayerPtr fakeQuantizeLayer, CNNLayerPtr scaleShift) const;

#if 0 // TODO LPT-TO-NGRAPH
    static Blob::Ptr reshapeWeightsIntervalConst(
        CNNLayer& constLayer,
        const std::vector<size_t>& dims,
        const Layout layout);

    static void reshapeFakeQuantize(
        CNNLayer& fakeQuantizeLayer,
        const std::vector<size_t>& dims,
        const Layout layout);
#endif
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph