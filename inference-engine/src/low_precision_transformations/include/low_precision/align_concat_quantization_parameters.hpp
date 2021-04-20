// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <low_precision/layer_transformation.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API AlignConcatQuantizationParamters;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

// Transformation creates `QuantizationAlignmentAttribute` attribute for FakeQuantize operations and
// forward propagate the attribute throught precision preserved operations. If `opset1::Convolution` operation is achieved then
// the transformation marks `QuantizationAlignmentAttribute` attribute as actual.
class ngraph::pass::low_precision::AlignConcatQuantizationParamters : public ngraph::pass::FunctionPass {
public:
    AlignConcatQuantizationParamters(LayerTransformation::Params params = LayerTransformation::Params());
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

protected:
    LayerTransformation::Params params;
};
