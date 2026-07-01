// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"
#include "openvino/opsets/opset1_decl.hpp"

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/strided_slice.hpp"
#include "openvino/op/strided_slice.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> StridedSliceFunction::get(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const std::vector<int64_t>& begin,
    const std::vector<int64_t>& end,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const std::vector<int64_t>& newAxisMask,
    const std::vector<int64_t>& shrinkAxisMask,
    const std::vector<int64_t>& elipsisMask,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto deq = makeDequantization(input, dequantization);

    const auto beginParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ begin.size() }, begin);
    beginParam->set_friendly_name("begin");
    const auto endParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ end.size() }, end);
    endParam->set_friendly_name("end");
    const auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ strides.size() }, strides);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice = std::make_shared<ov::opset1::StridedSlice>(
        deq, beginParam, endParam, stridesParam,
        beginMask, endMask, newAxisMask,
        shrinkAxisMask, elipsisMask);

    const auto output = makeDequantization(stridedSlice, dequantizationAfter);
    output->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ov::opset1::Result>(output);
    return std::make_shared<ov::Model>(
        ov::ResultVector{ res },
        ov::ParameterVector{ input },
        "StridedSliceTransformation");
}

std::shared_ptr<ov::Model> StridedSliceFunction::getWithParamInputs(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::PartialShape& beginShape,
    const ov::PartialShape& endShape,
    const ov::PartialShape& stridesShape,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto beginParam = std::make_shared<ov::opset1::Parameter>(ov::element::i64, beginShape);
    beginParam->set_friendly_name("begin");
    const auto endParam = std::make_shared<ov::opset1::Parameter>(ov::element::i64, endShape);
    endParam->set_friendly_name("end");
    const auto stridesParam = std::make_shared<ov::opset1::Parameter>(ov::element::i64, stridesShape);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice =
        std::make_shared<ov::opset1::StridedSlice>(deqBefore, beginParam, endParam, stridesParam, beginMask, endMask);

    const auto output = makeDequantization(stridedSlice, dequantizationAfter);
    output->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ov::opset1::Result>(output);
    return std::make_shared<ov::Model>(
        ov::ResultVector{ res },
        ov::ParameterVector{ input, beginParam, endParam, stridesParam },
        "StridedSliceTransformation");
}

std::shared_ptr<ov::Model> StridedSliceFunction::get(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
    const std::vector<int64_t>& begin,
    const std::vector<int64_t>& end,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& beginMask,
    const std::vector<int64_t>& endMask,
    const std::vector<int64_t>& newAxisMask,
    const std::vector<int64_t>& shrinkAxisMask,
    const std::vector<int64_t>& elipsisMask) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input");
    const auto fqOnData = makeFakeQuantize(input, inputPrecision, fakeQuantize);

    const auto beginParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ begin.size() }, begin);
    beginParam->set_friendly_name("begin");
    const auto endParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ end.size() }, end);
    endParam->set_friendly_name("end");
    const auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ strides.size() }, strides);
    stridesParam->set_friendly_name("strides");

    const auto stridedSlice = std::make_shared<ov::opset1::StridedSlice>(
        fqOnData, beginParam, endParam, stridesParam,
        beginMask, endMask, newAxisMask,
        shrinkAxisMask, elipsisMask);
    stridedSlice->set_friendly_name("StridedSlice");

    const auto res = std::make_shared<ov::opset1::Result>(stridedSlice);
    const auto function = std::make_shared<ov::Model>(
        ov::ResultVector{ res },
        ov::ParameterVector{ input },
        "StridedSliceTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
