// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <ngraph/ngraph.hpp>

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
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

private:
    void fillDequantization(
        std::shared_ptr<ngraph::Node> layer,
        std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
        std::vector<FakeQuantizeDequantization>& dequantizationsToConcatenate) const;

    void fillQuantization(
        const std::shared_ptr<ngraph::Node> layer,
        const std::unordered_map<std::string, FakeQuantizeDequantization>& dequantizationByFakeQuantize,
        std::vector<FakeQuantizeDequantization>& dequantization) const;

    FakeQuantizeDequantization getConcatenatedDequantization(
        const std::shared_ptr<ngraph::opset1::Concat> concat,
        const std::vector<FakeQuantizeDequantization>& dequantization) const;

    static FakeQuantizeDequantization getFoldedDequantization(
        const std::shared_ptr<ngraph::Node> operation,
        const FakeQuantizeDequantization& dequantization,
        const size_t sourceOutputIdx);

    static FakeQuantizeDequantization broadcastDequantiationConstant(const FakeQuantizeDequantization& deq);

    bool isMultiChannel(const std::vector<std::shared_ptr<ngraph::opset1::Concat>>& concatLayers) const noexcept;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
