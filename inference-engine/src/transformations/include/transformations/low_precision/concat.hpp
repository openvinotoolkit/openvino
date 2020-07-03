// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>
#include "layer_transformation.hpp"
#include "common/subgraph.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ConcatTransformation : public LayerTransformation {
public:
    ConcatTransformation(const Params& params) : LayerTransformation(params) {}
    ~ConcatTransformation() override {};
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

protected:
    void addDequantizationLayers(
        TransformationContext& context,
        ngraph::pass::low_precision::Subgraph& subgraph,
        std::function<void(
            ngraph::Node& layer,
            const std::string originalLayerName,
            std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate)> getLayerDequantizationCallback) const;

private:
    size_t getMinQuantizationLevels(
        const DataPrecision& dataPrecision,
        const float maxOutputInterval,
        const std::vector<QuantizationDetails>& quantizationLayersDetails,
        const float outputLowValue,
        const float outputHighValue) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
