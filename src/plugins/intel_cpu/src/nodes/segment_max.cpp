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

    // Update lastSegmentIds
    std::vector<int64_t> segmentIds(getSrcMemoryAtPort(1)->getSize());
    auto segmentIdsPrecision = getOriginalInputPrecisionAtPort(1);
    if (segmentIdsPrecision == ov::element::i32) {
        const int32_t* srcSegmentIds = getSrcDataAtPortAs<const int32_t>(1);
        std::copy(srcSegmentIds, srcSegmentIds + segmentIds.size(), segmentIds.begin());
    } else if (segmentIdsPrecision == ov::element::i64) {
        const int64_t* srcSegmentIds = getSrcDataAtPortAs<const int64_t>(1);
        std::copy(srcSegmentIds, srcSegmentIds + segmentIds.size(), segmentIds.begin());
    } else {
        OPENVINO_THROW("Unsupported index type for segment_ids input");
    }
    lastSegmentIds.assign(segmentIds.begin(), segmentIds.end());

    // Update lastNumSegments
    if (getOriginalInputsNumber() == 3) {
        const auto& numSegmentsPrecision = getOriginalInputPrecisionAtPort(2);
        int64_t numSegmentsValue;
        if (numSegmentsPrecision == ov::element::i64) {
            numSegmentsValue = *getSrcDataAtPortAs<const int64_t>(2);
        } else if (numSegmentsPrecision == ov::element::i32) {
            numSegmentsValue = static_cast<int64_t>(*getSrcDataAtPortAs<const int32_t>(2));
        } else {
            OPENVINO_THROW("Unsupported index type for num_segments input");
        }
        if (lastNumSegments.empty()) {
            lastNumSegments.push_back(numSegmentsValue);
        } else {
            lastNumSegments[0] = numSegmentsValue;
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
    auto segmentIdsPrecision = getOriginalInputPrecisionAtPort(1);
    const int64_t* segmentIds = nullptr;
    if (segmentIdsPrecision == ov::element::i32) {
        segmentIds = reinterpret_cast<const int64_t*>(getSrcDataAtPortAs<const int32_t>(1));
    } else if (segmentIdsPrecision == ov::element::i64) {
        segmentIds = getSrcDataAtPortAs<const int64_t>(1);
    } else {
        OPENVINO_THROW("Unsupported index type for segment_ids input");
    }
    if (lastSegmentIds.size() != getSrcMemoryAtPort(1)->getSize()) {
        return true;
    }
    for (size_t i = 0; i < lastSegmentIds.size(); i++) {
        if (lastSegmentIds[i] != segmentIds[i]) {
            return true;
        }
    }

    // Check if numSegments has changed
    if (getOriginalInputsNumber() == 3) {
        const auto& numSegmentsPrecision = getOriginalInputPrecisionAtPort(2);
        if (lastNumSegments.empty()) {
            return true;
        }
        if (numSegmentsPrecision == ov::element::i64) {
            const auto* numSegments = getSrcDataAtPortAs<const int64_t>(2);
            if (*numSegments != lastNumSegments[0]) {
                return true;
            }
        } else if (numSegmentsPrecision == ov::element::i32) {
            const auto* numSegments = getSrcDataAtPortAs<const int32_t>(2);
            if (static_cast<int64_t>(*numSegments) != lastNumSegments[0]) {
                return true;
            }
        } else {
            OPENVINO_THROW("Unsupported index type for num_segments input");
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
