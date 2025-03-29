// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/elementwise.hpp"

#include "low_precision/layer_transformation.hpp"
#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

namespace {

std::shared_ptr<ov::opset1::FakeQuantize> makeFakeQuantizeWithNames(
        const ov::Output<Node>& parent,
        const ov::element::Type precision,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::string name) {
    auto fq = ov::builder::subgraph::makeFakeQuantize(parent, precision, fqOnData);
    fq->set_friendly_name(name);
    fq->get_input_node_ptr(1)->set_friendly_name(name + "/inputLow");
    fq->get_input_node_ptr(2)->set_friendly_name(name + "/inputHigh");
    fq->get_input_node_ptr(3)->set_friendly_name(name + "/outputLow");
    fq->get_input_node_ptr(4)->set_friendly_name(name + "/outputHigh");
    return fq;
}

} // namespace

std::shared_ptr<ov::Model> ElementwiseFunction::getOriginalSubgraphWithConvolutions(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const bool broadcast,
        const std::string& elementWiseType,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore1,
        const ov::builder::subgraph::Convolution& convolution1,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter1,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore2,
        const ov::builder::subgraph::Convolution& convolution2,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter2,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter) {
    ov::PartialShape inputShape2 = inputShape;

    if (broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto makeBranch = [&](
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const size_t index,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataBefore,
        const ov::builder::subgraph::Convolution& convolution,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnDataAfter) ->
            std::pair<std::shared_ptr<ov::opset1::Parameter>, std::shared_ptr<ov::Node>> {
        const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input->set_friendly_name("input" + std::to_string(index));
        std::shared_ptr<ov::Node> parent = input;

        if (!fqOnDataBefore.empty()) {
            parent = makeFakeQuantizeWithNames(parent, precision, fqOnDataBefore, "fakeQuantizeBefore" + std::to_string(index));
        }

        if (!convolution.empty()) {
            parent = makeConvolution(parent, convolution);
            parent->set_friendly_name("convolution" + std::to_string(index));
        }

        if (!fqOnDataAfter.empty()) {
            parent = makeFakeQuantizeWithNames(parent, precision, fqOnDataAfter, "fakeQuantizeAfter" + std::to_string(index));
        }

        return std::make_pair(input, parent);
    };

    const auto branch1 = makeBranch(precision, inputShape, 1, fqOnDataBefore1, convolution1, fqOnDataAfter1);
    const auto branch2 = makeBranch(precision, inputShape, 2, fqOnDataBefore2, convolution2, fqOnDataAfter2);

    std::shared_ptr<ov::Node> result;
    if (elementWiseType == "add") {
        result = std::make_shared<ov::opset1::Add>(branch1.second, branch2.second);
        result->set_friendly_name("add");
    } else if (elementWiseType == "multiply") {
        result = std::make_shared<ov::opset1::Multiply>(branch1.second, branch2.second);
        result->set_friendly_name("multiply");
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "not supported element-wise operation type " << elementWiseType;
    }

    if (!fqOnDataAfter.empty()) {
        result = makeFakeQuantizeWithNames(result, precision, fqOnDataAfter, "fakeQuantizeAfter");

        // we need a some operation to move dequantization operations away from FakeQuantize to avoid cleanup fuse
        result = std::make_shared<ov::opset1::MaxPool>(result,
                                                       Strides{1, 1},
                                                       Shape{1, 1},
                                                       Shape{0, 0},
                                                       Shape{2, 2},
                                                       ov::op::RoundingType::FLOOR);
        result->set_friendly_name("maxPool");
    }

    result = std::make_shared<ov::opset1::Result>(result);
    result->set_friendly_name("result");

    ov::ResultVector results{ ov::as_type_ptr<ov::opset1::Result>(result) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ branch1.first, branch2.first }, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
