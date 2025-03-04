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
    ov::element::Type segmentIdsPrecision = getOriginalInputPrecisionAtPort(1);
    if (getOriginalInputsNumber() == 2) {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, segmentIdsPrecision}},
                             {{LayoutType::ncsp, dataPrecision}},
                             impl_desc_type::ref);
    } else {
        ov::element::Type numSegmentsPrecision = getOriginalInputPrecisionAtPort(2);
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, segmentIdsPrecision},
                              {LayoutType::ncsp, numSegmentsPrecision}},
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
    if (getOriginalInputsNumber() == 3) {
        const int64_t* numSegments = getSrcDataAtPortAs<const int64_t>(2);
        if (lastNumSegments.empty()) {
            lastNumSegments.push_back(*numSegments);
        } else {
            lastNumSegments[0] = *numSegments;
        }
    }
}

bool SegmentMax::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (getOriginalInputsNumber() == 3) {
        if (lastNumSegments.empty()) {
            return true;
        }
        const int64_t* numSegments = getSrcDataAtPortAs<const int64_t>(2);
        if (*numSegments != lastNumSegments[0]) {
            return true;
        }
    }
    return false;
}

template <class T, class T_idx>
void SegmentMax::executeImpl() {
    const auto& data_shape = getSrcMemoryAtPort(0)->getStaticDims();
    const auto& output_shape = getDstMemoryAtPort(0)->getShape().getStaticDims();
    const auto empty_segment_value = fillMode == ov::op::FillMode::ZERO ? T(0) : std::numeric_limits<T>::lowest();
    ov::reference::segment_max(getSrcDataAtPortAs<const T>(0),
                               data_shape,
                               getSrcDataAtPortAs<const T_idx>(1),
                               getDstDataAtPortAs<T>(0),
                               output_shape,
                               empty_segment_value);
}

namespace {
struct SegmentMaxContext {
    SegmentMax& node;
};
}  // namespace

template <typename T>
struct SegmentMax::SegmentMaxExecute {
    using TData = typename std::tuple_element<0, T>::type;
    using TIndex = typename std::tuple_element<1, T>::type;
    void operator()(SegmentMaxContext& ctx) {
        ctx.node.executeImpl<TData, TIndex>();
    }
};

void SegmentMax::execute(const dnnl::stream& strm) {
    auto dataPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto indicesPrecision = getParentEdgeAt(1)->getMemory().getDesc().getPrecision();
    SegmentMaxContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              SegmentMaxExecute,
              ctx,
              std::tie(dataPrecision, indicesPrecision),
              OV_CASE2(ov::element::f32, ov::element::i32, float, int32_t),
              OV_CASE2(ov::element::f32, ov::element::i64, float, int64_t),
              OV_CASE2(ov::element::f16, ov::element::i32, ov::float16, int32_t),
              OV_CASE2(ov::element::f16, ov::element::i64, ov::float16, int64_t),
              OV_CASE2(ov::element::bf16, ov::element::i32, ov::bfloat16, int32_t),
              OV_CASE2(ov::element::bf16, ov::element::i64, ov::bfloat16, int64_t),
              OV_CASE2(ov::element::i8, ov::element::i32, int8_t, int32_t),
              OV_CASE2(ov::element::i8, ov::element::i64, int8_t, int64_t),
              OV_CASE2(ov::element::u8, ov::element::i32, uint8_t, int32_t),
              OV_CASE2(ov::element::u8, ov::element::i64, uint8_t, int64_t),
              OV_CASE2(ov::element::i32, ov::element::i32, int32_t, int32_t),
              OV_CASE2(ov::element::i32, ov::element::i64, int32_t, int64_t),
              OV_CASE2(ov::element::i64, ov::element::i32, int64_t, int32_t),
              OV_CASE2(ov::element::i64, ov::element::i64, int64_t, int64_t))
}
}  // namespace ov::intel_cpu::node
