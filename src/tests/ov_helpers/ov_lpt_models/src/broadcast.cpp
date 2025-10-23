// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/broadcast.hpp"

#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "openvino/op/broadcast.hpp"

namespace ov {
namespace builder {
namespace subgraph {

namespace {
template <typename T>
std::shared_ptr<ov::Node> make_broadcast(const std::shared_ptr<ov::Node>& parent,
                                         const Shape& targetShape,
                                         const Shape& axesMapping) {
    if (axesMapping.empty()) {
        return std::make_shared<T>(
            parent,
            std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{targetShape.size()}, targetShape));
    }
    return std::make_shared<T>(
        parent,
        std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{targetShape.size()}, targetShape),
        std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{axesMapping.size()}, axesMapping));
}
} // namespace

std::shared_ptr<ov::Model> BroadcastFunction::get(
    const bool v1,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const Shape& targetShape,
    const Shape& axesMapping,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    const auto dq_before = makeDequantization(input, dequantizationBefore);

    const auto bcast = v1 ? make_broadcast<ov::opset1::Broadcast>(dq_before, targetShape, axesMapping)
                          : make_broadcast<ov::opset3::Broadcast>(dq_before, targetShape, axesMapping);
    bcast->set_friendly_name("broadcast");

    const auto dq_after = makeDequantization(bcast, dequantizationAfter);
    const auto model =
        std::make_shared<ov::Model>(ov::OutputVector{dq_after}, ov::ParameterVector{input}, "BroadcastTransformation");
    return model;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
