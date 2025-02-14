// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements.h"

#include <cmath>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool GatherElements::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v6::GatherElements::get_type_info_static())) {
            errorMessage = "Node is not an instance of the GatherElements operation from operation set v6.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GatherElements::GatherElements(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (inputShapes.size() != 2 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has invalid number of input/output edges.");
    }

    const auto dataRank = getInputShapeAtPort(dataIndex_).getRank();
    const auto indicesRank = getInputShapeAtPort(indicesIndex_).getRank();
    if (dataRank != indicesRank) {
        THROW_CPU_NODE_ERR("has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.");
    }

    auto gatherElementsOp = ov::as_type_ptr<ov::op::v6::GatherElements>(op);
    auto axis = gatherElementsOp->get_axis();
    if (axis < 0) {
        axis += dataRank;
    }
    if (axis < 0 || axis >= static_cast<int>(dataRank)) {
        THROW_CPU_NODE_ERR("has invalid axis attribute: ", axis);
    }
    axis_ = axis;
}

void GatherElements::prepareParams() {
    const auto& dataDims = getParentEdgeAt(dataIndex_)->getMemory().getStaticDims();
    const auto& dstDims = getChildEdgeAt(0)->getMemory().getStaticDims();
    strideAxDst_ = 1;
    for (size_t i = dstDims.size() - 1; i > axis_; i--) {
        strideAxDst_ *= dstDims[i];
    }
    dstAxDim_ = dstDims[axis_];
    if (axis_ > 0) {
        strideAx1Diff_ = 1;
        for (size_t i = dataDims.size() - 1; i >= axis_; i--) {
            strideAx1Diff_ *= dataDims[i];
        }
        strideAx1Diff_ -= strideAxDst_ * dstDims[axis_];
    }
}

void GatherElements::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(dataIndex_);
    if (!one_of(inDataPrecision.size(),
                sizeof(element_type_traits<ov::element::i32>::value_type),
                sizeof(element_type_traits<ov::element::i16>::value_type),
                sizeof(element_type_traits<ov::element::i8>::value_type))) {
        THROW_CPU_NODE_ERR("has unsupported 'inputData' input precision: ", inDataPrecision);
    }

    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(indicesIndex_);
    if (!one_of(indicesPrecision, ov::element::i32, ov::element::i64)) {
        THROW_CPU_NODE_ERR("has unsupported 'indices' input precision: ", indicesPrecision);
    }

    dataTypeSize_ = inDataPrecision.size();

    addSupportedPrimDesc({{LayoutType::ncsp, inDataPrecision}, {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

void GatherElements::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

template <typename dataType>
void GatherElements::directExecution() {
    const auto* srcData = getSrcDataAtPortAs<const dataType>(dataIndex_);
    const auto* indices = getSrcDataAtPortAs<const int>(indicesIndex_);
    auto* dstData = getDstDataAtPortAs<dataType>(0);

    const int outSize = getChildEdgeAt(0)->getMemory().getShape().getElementsCount();
    auto threadBody = [&](const int ithr, const int nthr) {
        int start(0lu), end(0lu);
        splitter(outSize, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        int axStrideIt = start % strideAxDst_;
        int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
        int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;

        for (int o = start; o < end; o++, axStrideIt++) {
            if (axStrideIt == strideAxDst_) {
                axStrideIt = 0;
                dstAxIdx++;
                if (dstAxIdx == dstAxDim_) {
                    dstAxIdx = 0;
                    dstShift0 += strideAx1Diff_;
                }
            }
            dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
        }
    };

    parallel_nt(0, threadBody);
}

void GatherElements::execute(const dnnl::stream& strm) {
    switch (dataTypeSize_) {
    case sizeof(element_type_traits<ov::element::i32>::value_type):
        return directExecution<element_type_traits<ov::element::i32>::value_type>();
    case sizeof(element_type_traits<ov::element::i16>::value_type):
        return directExecution<element_type_traits<ov::element::i16>::value_type>();
    case sizeof(element_type_traits<ov::element::i8>::value_type):
        return directExecution<element_type_traits<ov::element::i8>::value_type>();
    default:
        THROW_CPU_NODE_ERR("Unsupported data type size");
    }
}

bool GatherElements::created() const {
    return getType() == Type::GatherElements;
}

}  // namespace ov::intel_cpu::node
