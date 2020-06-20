// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const FakeQuantizeOnData& fakeQuantizeOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
