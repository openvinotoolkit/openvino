// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_range_node.h"
#include <utils/general_utils.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNRangeNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v0::Range::get_type_info_static(), ngraph::op::v4::Range::get_type_info_static())) {
            errorMessage = "Only opset1 and opset4 Range operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNRangeNode::MKLDNNRangeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Range layer with name '" + op->get_friendly_name() + "'";

    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    SizeVector start_dims = op->get_input_shape(RANGE_START);
    if (ngraph::shape_size(start_dims) != 1)
        IE_THROW() << errorPrefix << " has start scalar with more than 1 value";

    SizeVector limit_dims = op->get_input_shape(RANGE_LIMIT);
    if (ngraph::shape_size(limit_dims) != 1)
        IE_THROW() << errorPrefix << " has limit scalar with more than 1 value";

    SizeVector delta_dims = op->get_input_shape(RANGE_DELTA);
    if (ngraph::shape_size(delta_dims) != 1)
        IE_THROW() << errorPrefix << " has delta scalar with more than 1 value";

    size_t dstRank = op->get_output_partial_shape(0).size();
    if (dstRank > 1)
        IE_THROW() << errorPrefix << " has unsupported rank for output: " << dstRank;
}

void MKLDNNRangeNode::initSupportedPrimitiveDescriptors() {
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
        for (int i = 0; i < inputShapes.size(); ++i)
            inDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    } else {
        inDataConf.reserve(inputShapes.size());
        for (int i = 0; i < inputShapes.size(); ++i)
            inDataConf.emplace_back(LayoutType::ncsp);
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    }
}

std::vector<VectorDims> MKLDNNRangeNode::shapeInfer() const {
    return MKLDNNNode::shapeInferGeneric(PortMask(RANGE_START, RANGE_LIMIT, RANGE_DELTA));
}

void MKLDNNRangeNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void MKLDNNRangeNode::execute(mkldnn::stream strm) {
    StatusCode retcode = OK;
    switch (getParentEdgeAt(0)->getMemory().getDesc().getPrecision()) {
        case Precision::FP32:
            retcode = rangeKernel<float>();
            break;
        case Precision::I32:
            retcode = rangeKernel<int32_t>();
            break;
        default:
            IE_THROW() << "Incorrect output precision. Only FP32 and I32 are supported!";
    }
    if (retcode == PARAMETER_MISMATCH) {
        std::string errorMsg = "Range indexes exceeds data tensor dimension";
        IE_THROW() << errorMsg;
    }
}

template <typename data_t>
size_t MKLDNNRangeNode::getWorkAmount(data_t *startPtr, data_t *stopPtr, data_t *stepPtr) const {
    data_t start = 0, limit = 0, delta = 0;
    if (startPtr == nullptr)
        startPtr = &start;
    if (stopPtr == nullptr)
        stopPtr = &limit;
    if (stepPtr == nullptr)
        stepPtr = &delta;
    *startPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_START)->getMemoryPtr()->GetPtr())[0];
    *stopPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_LIMIT)->getMemoryPtr()->GetPtr())[0];
    *stepPtr = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_DELTA)->getMemoryPtr()->GetPtr())[0];
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
InferenceEngine::StatusCode MKLDNNRangeNode::rangeKernel() {
    data_t start = 0, delta = 0;
    size_t work_amount_dst = getWorkAmount<data_t>(&start, nullptr, &delta);
    if (isDynamicNode()) {
        VectorDims newOutputShape {work_amount_dst};
        redefineOutputMemory({newOutputShape});
    }
    data_t* dst_data = reinterpret_cast<data_t *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
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
bool MKLDNNRangeNode::created() const {
    return getType() == Range;
}

REG_MKLDNN_PRIM_FOR(MKLDNNRangeNode, Range)
