// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_elements_node.h"
#include <ngraph/opsets/opset1.hpp>
#include <precision_utils.h>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherElementsNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v6::GatherElements>(op);
        if (!gatherElementsOp) {
            errorMessage = "Node is not an instance of the GatherElements operation from operation set v6.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherElementsNode::MKLDNNGatherElementsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix_ = std::string("Layer GatherElements with name '") + op->get_friendly_name() + "'";

    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix_ << " has invalid number of input/output edges.";

    const auto& dataDims = op->get_input_shape(dataIndex_);
    const auto& indicesDims = op->get_input_shape(indicesIndex_);
    if (dataDims.size() != indicesDims.size())
        IE_THROW() << errorPrefix_ << " has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.";

    auto gatherElementsOp = ngraph::as_type_ptr<const ngraph::op::v6::GatherElements>(op);
    auto axis = gatherElementsOp->get_axis();
    if (axis < 0)
        axis += dataDims.size();
    if (axis < 0 || axis >= static_cast<int>(dataDims.size()))
        IE_THROW() << errorPrefix_ << " has invalid axis attribute: " << axis;
    axis_ = axis;

    auto outputShape = op->get_output_shape(0);
    strideAxDst_ = 1;
    for (int i = outputShape.size() - 1; i > axis_; i--)
        strideAxDst_ *= outputShape[i];
    dstAxDim_ = op->get_output_shape(0)[axis_];
    if (axis_ > 0) {
        strideAx1Diff_ = 1;
        for (int i = dataDims.size() - 1; i >= axis_; i--)
            strideAx1Diff_ *= dataDims[i];
        strideAx1Diff_ -= strideAxDst_ * outputShape[axis_];
    }
}

void MKLDNNGatherElementsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(dataIndex_);
    if (!MKLDNNPlugin::one_of(inDataPrecision.size(),
                              sizeof(PrecisionTrait<Precision::I32>::value_type),
                              sizeof(PrecisionTrait<Precision::I16>::value_type),
                              sizeof(PrecisionTrait<Precision::I8>::value_type))) {
        IE_THROW() << errorPrefix_ << " has unsupported 'inputData' input precision: " << inDataPrecision;
    }

    Precision indicesPrecision = getOriginalInputPrecisionAtPort(indicesIndex_);
    if (!MKLDNNPlugin::one_of(indicesPrecision, Precision::I32, Precision::I64)) {
        IE_THROW() << errorPrefix_ << " has unsupported 'indices' input precision: " << indicesPrecision;
    }

    dataTypeSize_ = inDataPrecision.size();

    addSupportedPrimDesc({{GeneralLayout::ncsp, inDataPrecision},
                          {GeneralLayout::ncsp, Precision::I32}},
                         {{GeneralLayout::ncsp, inDataPrecision}},
                         impl_desc_type::ref_any);
}

template <typename dataType>
void MKLDNNGatherElementsNode::directExecution() {
    const auto *srcData = reinterpret_cast<const dataType *>(getParentEdgeAt(dataIndex_)->getMemoryPtr()->GetPtr());
    const auto *indices = reinterpret_cast<const int *>(getParentEdgeAt(indicesIndex_)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<dataType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const int outSize = getChildEdgeAt(0)->getShape().getElementsCount();
    auto threadBody = [&](const int ithr, const int nthr) {
        int start(0lu), end(0lu);
        splitter(outSize, nthr, ithr, start, end);
        if (start >= end)
            return;

        int axStrideIt = start % strideAxDst_;
        int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
        int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;

        for (size_t o = start; o < end; o++, axStrideIt++) {
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

void MKLDNNGatherElementsNode::execute(mkldnn::stream strm) {
    switch (dataTypeSize_) {
        case sizeof(PrecisionTrait<Precision::I32>::value_type):
            return directExecution<PrecisionTrait<Precision::I32>::value_type>();
        case sizeof(PrecisionTrait<Precision::I16>::value_type):
            return directExecution<PrecisionTrait<Precision::I16>::value_type>();
        case sizeof(PrecisionTrait<Precision::I8>::value_type):
            return directExecution<PrecisionTrait<Precision::I8>::value_type>();
        default:
            return IE_THROW() << "Unsupported data type size";
    }
}

bool MKLDNNGatherElementsNode::created() const {
    return getType() == GatherElements;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherElementsNode, GatherElements)
