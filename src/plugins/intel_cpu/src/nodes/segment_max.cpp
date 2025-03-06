// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max.h"

#include "openvino/op/segment_max.hpp"
#include "openvino/reference/segment_max.hpp"

namespace ov::intel_cpu::node {
SegmentMax::SegmentMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    fillMode = ov::as_type_ptr<const ov::op::v16::SegmentMax>(op)->get_fill_mode();
}

bool SegmentMax::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v16::SegmentMax>(op)) {
            errorMessage = "Only opset16 SegmentMax operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void SegmentMax::getSupportedDescriptors() {
    // Validation is already done in the ov::opset16::SegmentMax
}

void SegmentMax::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (getOriginalInputsNumber() == 2) {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, ov::element::i32}},
                             {{LayoutType::ncsp, dataPrecision}},
                             impl_desc_type::ref);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32}},
                             {{LayoutType::ncsp, dataPrecision}},
                             impl_desc_type::ref);
    }
}

bool SegmentMax::created() const {
    return getType() == Type::SegmentMax;
}

bool SegmentMax::needPrepareParams() const {
    return false;
}

void SegmentMax::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    // Update lastSegmentIds
    const auto* srcSegmentIds = getSrcDataAtPortAs<const int32_t>(1);
    lastSegmentIds.assign(srcSegmentIds, srcSegmentIds + getSrcMemoryAtPort(1)->getSize());

    // Update lastNumSegments
    if (getOriginalInputsNumber() == 3) {
        const auto* numSegmentsValue = getSrcDataAtPortAs<const int32_t>(2);
        if (lastNumSegments.empty()) {
            lastNumSegments.push_back(*numSegmentsValue);
        } else {
            lastNumSegments[0] = *numSegmentsValue;
        }
    }
}

bool SegmentMax::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (lastSegmentIds.empty()) {
        return true;
    }

    // Check if segmentIds has changed
    if (lastSegmentIds.size() != getSrcMemoryAtPort(1)->getSize()) {
        return true;
    }
    const auto* segmentIds = getSrcDataAtPortAs<const int32_t>(1);
    for (size_t i = 0; i < lastSegmentIds.size(); i++) {
        if (lastSegmentIds[i] != segmentIds[i]) {
            return true;
        }
    }

    // Check if numSegments has changed
    if (getOriginalInputsNumber() == 3) {
        if (lastNumSegments.empty()) {
            return true;
        }
        const auto* numSegments = getSrcDataAtPortAs<const int32_t>(2);
        if (*numSegments != lastNumSegments[0]) {
            return true;
        }
    }
    return false;
}

template <class T>
void SegmentMax::executeImpl() {
    const auto& data_shape = getSrcMemoryAtPort(0)->getStaticDims();
    const auto& output_shape = getDstMemoryAtPort(0)->getShape().getStaticDims();
    const auto empty_segment_value = fillMode == ov::op::FillMode::ZERO ? T(0) : std::numeric_limits<T>::lowest();
    ov::reference::segment_max(getSrcDataAtPortAs<const T>(0),
                               data_shape,
                               getSrcDataAtPortAs<const int32_t>(1),
                               getDstDataAtPortAs<T>(0),
                               output_shape,
                               empty_segment_value);
}

namespace {
struct SegmentMaxContext {
    SegmentMax& node;
};
}  // namespace

template <class T>
struct SegmentMax::SegmentMaxExecute {
    void operator()(SegmentMaxContext& ctx) {
        ctx.node.executeImpl<T>();
    }
};

void SegmentMax::execute(const dnnl::stream& strm) {
    auto dataPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    SegmentMaxContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              SegmentMaxExecute,
              ctx,
              dataPrecision,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::bf16, ov::bfloat16),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i64, int64_t))
}
}  // namespace ov::intel_cpu::node
