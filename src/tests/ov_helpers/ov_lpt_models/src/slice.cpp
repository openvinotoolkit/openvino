// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/slice.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> SliceFunction::get(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
    const std::vector<int64_t>& start,
    const std::vector<int64_t>& stop,
    const std::vector<int64_t>& step,
    const std::vector<int64_t>& axes) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");

    std::shared_ptr<ov::Node> parent = input;
    if (!fakeQuantize.empty()) {
        parent = makeFakeQuantize(parent, inputPrecision, fakeQuantize);
    }

    const auto start_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ start.size() }, start);
    start_constant->set_friendly_name("start");
    const auto stop_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ stop.size() }, stop);
    stop_constant->set_friendly_name("stop");
    const auto step_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ step.size() }, step);
    step_constant->set_friendly_name("step");
    const auto axes_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes);
    axes_constant->set_friendly_name("axes ");

    const auto stridedSlice = std::make_shared<ov::opset8::Slice>(
        parent,
        start_constant,
        stop_constant,
        step_constant,
        axes_constant);
    stridedSlice->set_friendly_name("slice");

    const auto res = std::make_shared<ov::opset1::Result>(stridedSlice);
    const auto function = std::make_shared<ov::Model>(
        ov::ResultVector{ res },
        ov::ParameterVector{ input },
        "SliceTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
