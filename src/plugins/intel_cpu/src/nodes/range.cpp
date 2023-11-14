// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <openvino/opsets/opset1.hpp>
#include "openvino/core/parallel.hpp"
#include "range.h"
#include <utils/general_utils.h>
#include <shape_inference/shape_inference_internal_dyn.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Range::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v0::Range::get_type_info_static(), ov::op::v4::Range::get_type_info_static())) {
            errorMessage = "Only opset1 and opset4 Range operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Range::Range(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = "Range layer with name '" + op->get_friendly_name() + "'";

    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input/output edges!");

    SizeVector start_dims = op->get_input_shape(RANGE_START);
    if (ov::shape_size(start_dims) != 1)
        OPENVINO_THROW(errorPrefix, " has start scalar with more than 1 value");

    SizeVector limit_dims = op->get_input_shape(RANGE_LIMIT);
    if (ov::shape_size(limit_dims) != 1)
        OPENVINO_THROW(errorPrefix, " has limit scalar with more than 1 value");

    SizeVector delta_dims = op->get_input_shape(RANGE_DELTA);
    if (ov::shape_size(delta_dims) != 1)
        OPENVINO_THROW(errorPrefix, " has delta scalar with more than 1 value");

    size_t dstRank = op->get_output_partial_shape(0).size();
    if (dstRank > 1)
        OPENVINO_THROW(errorPrefix, " has unsupported rank for output: ", dstRank);
}

void Range::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    if (!(getOriginalInputPrecisionAtPort(RANGE_START) == Precision::I32 &&
            getOriginalInputPrecisionAtPort(RANGE_LIMIT) == Precision::I32 &&
            getOriginalInputPrecisionAtPort(RANGE_DELTA) == Precision::I32 &&
            getOriginalOutputPrecisionAtPort(0)     == Precision::I32) &&
        !(getOriginalInputPrecisionAtPort(RANGE_START) == Precision::FP32 &&
            getOriginalInputPrecisionAtPort(RANGE_LIMIT) == Precision::FP32 &&
            getOriginalInputPrecisionAtPort(RANGE_DELTA) == Precision::FP32 &&
            getOriginalOutputPrecisionAtPort(0) == Precision::FP32)) {
        inDataConf.reserve(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); ++i)
            inDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    } else {
        inDataConf.reserve(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); ++i)
            inDataConf.emplace_back(LayoutType::ncsp);
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    }
}

void Range::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Range::execute(dnnl::stream strm) {
    StatusCode retcode = OK;
    switch (getParentEdgeAt(0)->getMemory().getDesc().getPrecision()) {
        case Precision::FP32:
            retcode = rangeKernel<float>();
            break;
        case Precision::I32:
            retcode = rangeKernel<int32_t>();
            break;
        default:
            OPENVINO_THROW("Incorrect output precision. Only FP32 and I32 are supported!");
    }
    if (retcode == PARAMETER_MISMATCH) {
        std::string errorMsg = "Range indexes exceeds data tensor dimension";
        OPENVINO_THROW(errorMsg);
    }
}

template <typename data_t>
size_t Range::getWorkAmount(data_t *startPtr, data_t *stopPtr, data_t *stepPtr) const {
    data_t start = 0, limit = 0, delta = 0;
    if (startPtr == nullptr)
        startPtr = &start;
    if (stopPtr == nullptr)
        stopPtr = &limit;
    if (stepPtr == nullptr)
        stepPtr = &delta;
    *startPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_START)->getMemoryPtr()->getData())[0];
    *stopPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_LIMIT)->getMemoryPtr()->getData())[0];
    *stepPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_DELTA)->getMemoryPtr()->getData())[0];
    const data_t span = *stopPtr - *startPtr;
    const data_t step = *stepPtr;
    if (std::is_same<data_t, int>::value) {
        int iSpan = static_cast<int>(span);
        int iStep = static_cast<int>(step);
        return static_cast<size_t>(div_up(iSpan < 0 ? -iSpan : iSpan, iStep < 0 ? -iStep : iStep));
    } else {
        return static_cast<size_t>(std::ceil(std::fabs(span) / std::fabs(step)));
    }
}

template <typename data_t>
InferenceEngine::StatusCode Range::rangeKernel() {
    data_t start = 0, delta = 0;
    size_t work_amount_dst = getWorkAmount<data_t>(&start, nullptr, &delta);
    if (isDynamicNode()) {
        VectorDims newOutputShape {work_amount_dst};
        redefineOutputMemory({newOutputShape});
    }
    data_t* dst_data = reinterpret_cast<data_t *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->getData());
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t iwork = 0, end = 0;
        splitter(work_amount_dst, nthr, ithr, iwork, end);
        data_t dst_value = start + iwork * delta;
        for (; iwork < end; ++iwork, dst_value += delta) {
            dst_data[iwork] = dst_value;
        }
    });
    return OK;
}

bool Range::created() const {
    return getType() == Type::Range;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
