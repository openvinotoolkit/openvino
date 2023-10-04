// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/space_to_batch.hpp"

#include <ngraph/opsets/opset2.hpp>
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SpaceToBatchFunction::get(const ngraph::PartialShape& input_shape,
                                                            const ngraph::element::Type input_type,
                                                            const FakeQuantizeOnData& fq_on_data,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& pads_begin,
                                                            const std::vector<size_t>& pads_end) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = fq_on_data.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, input_type, fq_on_data);

    parent = std::make_shared<ngraph::opset2::SpaceToBatch>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ pads_begin.size() }, pads_begin),
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ pads_end.size() }, pads_end));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(parent)};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SpaceToBatchFunction");
}

std::shared_ptr<ngraph::Function> SpaceToBatchFunction::get(const ngraph::PartialShape& input_shape,
                                                            const ngraph::element::Type input_type,
                                                            const ngraph::builder::subgraph::DequantizationOperations& dequantization_before,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& pads_begin,
                                                            const std::vector<size_t>& pads_end,
                                                            const ngraph::builder::subgraph::DequantizationOperations& dequantization_after) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = dequantization_before.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeDequantization(input, dequantization_before);

    parent = std::make_shared<ngraph::opset2::SpaceToBatch>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ pads_begin.size() }, pads_begin),
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ pads_end.size() }, pads_end));

    parent = makeDequantization(parent, dequantization_after);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(parent)};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SpaceToBatchFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
