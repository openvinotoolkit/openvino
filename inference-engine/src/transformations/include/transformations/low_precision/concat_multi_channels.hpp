// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>

#include "concat.hpp"
#include "common/subgraph.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ConcatMultiChannelsTransformation : public ConcatTransformation {
public:
    ConcatMultiChannelsTransformation(const Params& params) : ConcatTransformation(params) {}
    ~ConcatMultiChannelsTransformation() override {};
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

private:
    static void fillDequantization(
        ngraph::Node& layer,
        const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate);

    static void fillQuantization(const ngraph::Node& layer, std::vector<ngraph::opset1::FakeQuantize*>& fakeQuantizes);
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
