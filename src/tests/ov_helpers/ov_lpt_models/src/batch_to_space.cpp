// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/batch_to_space.hpp"

#include <openvino/opsets/opset2.hpp>
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> BatchToSpaceFunction::get(const ov::PartialShape& input_shape,
                                                            const ov::element::Type input_type,
                                                            const FakeQuantizeOnData& fq_on_data,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& crops_begin,
                                                            const std::vector<size_t>& crops_end) {
    const auto input = std::make_shared<ov::opset1::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = fq_on_data.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, input_type, fq_on_data);

    parent = std::make_shared<ov::opset2::BatchToSpace>(
        parent,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ crops_begin.size() }, crops_begin),
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ crops_end.size() }, crops_end));

    ov::ResultVector results{std::make_shared<ov::opset1::Result>(parent)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "BatchToSpaceFunction");
}

std::shared_ptr<ov::Model> BatchToSpaceFunction::get(const ov::PartialShape& input_shape,
                                                            const ov::element::Type input_type,
                                                            const ngraph::builder::subgraph::DequantizationOperations& dequantization_before,
                                                            const std::vector<size_t>& block_shape,
                                                            const std::vector<size_t>& crops_begin,
                                                            const std::vector<size_t>& crops_end,
                                                            const ngraph::builder::subgraph::DequantizationOperations& dequantization_after) {
    const auto input = std::make_shared<ov::opset1::Parameter>(input_type, input_shape);

    std::shared_ptr<Node> parent = dequantization_before.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeDequantization(input, dequantization_before);

    parent = std::make_shared<ov::opset2::BatchToSpace>(
        parent,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ block_shape.size() }, block_shape),
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ crops_begin.size() }, crops_begin),
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ crops_end.size() }, crops_end));

    parent = makeDequantization(parent, dequantization_after);

    ov::ResultVector results{std::make_shared<ov::opset1::Result>(parent)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "BatchToSpaceFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
