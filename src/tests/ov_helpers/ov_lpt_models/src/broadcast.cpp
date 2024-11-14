// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/broadcast.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

namespace {
template <typename T>
std::shared_ptr<ov::Node> make_broadcast(const std::shared_ptr<ov::Node>& parent, const Shape& tagetShape, const Shape& axesMapping) {
    return std::make_shared<T>(
        parent,
        std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{ tagetShape.size() }, tagetShape),
        std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{ axesMapping.size() }, axesMapping));
}
} // namespace

std::shared_ptr<ov::Model> BroadcastFunction::get(
    const bool v1,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const Shape& tagetShape,
    const Shape& axesMapping,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ov::Node> parent = input;

    if (!dequantizationBefore.empty()) {
        parent = makeDequantization(parent, dequantizationBefore);
    }

    parent = v1 ?
        make_broadcast<ov::opset1::Broadcast>(parent, tagetShape, axesMapping) :
        make_broadcast<ov::opset3::Broadcast>(parent, tagetShape, axesMapping);
    parent->set_friendly_name("broadcast");

    if (!dequantizationAfter.empty()) {
        parent = makeDequantization(parent, dequantizationAfter);
    }

    const std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(parent);

    const std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "BroadcastTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
