// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include <ngraph/opsets/opset5.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_log_softmax_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNLogSoftmaxNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto logSoftMax = std::dynamic_pointer_cast<const ngraph::opset5::LogSoftmax>(op);
        if (!logSoftMax) {
            errorMessage = "Only opset5 LogSoftmax operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNLogSoftmaxNode::MKLDNNLogSoftmaxNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "LogSoftmax layer with name '" + op->get_friendly_name() + "'";
    const auto logSoftMax = std::dynamic_pointer_cast<const ngraph::opset5::LogSoftmax>(op);
    if (logSoftMax == nullptr)
        IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
            "' is not an instance of LogSoftmax from opset5.";

    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    SizeVector dims = getInputShapeAtPort(0).getDims();
    if (dims.empty())
        dims = SizeVector(1, 1);
    axis = logSoftMax->get_axis();
    if (axis < 0)
        axis += dims.size();

    if (dims.size() < static_cast<size_t>((size_t)(1) + axis))
        IE_THROW() << errorPrefix << " has incorrect input parameters dimensions and axis number!";

    int j;
    for (j = dims.size() - 1; j >= 0; j--) {
        if (dims[j] != Shape::UNDEFINED_DIM && dims[j] != 1) break;
    }
    if (dims[j] != Shape::UNDEFINED_DIM && j == axis) isLastDim = true;
}

void MKLDNNLogSoftmaxNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32}},
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNLogSoftmaxNode::execute(mkldnn::stream strm) {
    const float *srcData = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    float* dstData = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    const auto dims = getParentEdgesAtPort(0)[0]->getMemory().getStaticDims();
    size_t reducedAxisSize;
    size_t reducedAxisStride = 1;
    size_t axisStep = 1;

    for (int i = 0; i < axis; i++)
        axisStep *= dims[i];
    reducedAxisSize = dims[axis];
    for (size_t i = (axis + 1); i < dims.size(); i++)
        reducedAxisStride *= dims[i];

    if (isLastDim) {
        parallel_for(axisStep, [&](size_t i) {
            const float *srcDataPtr = &srcData[i * reducedAxisSize];
            float *dstDataPtr = &dstData[i * reducedAxisSize];

            float reduceProd = 0.0f;
            const float max = *std::max_element(srcDataPtr, srcDataPtr + reducedAxisSize);
            for (size_t j = 0; j < reducedAxisSize; ++j)
                reduceProd += expf(srcDataPtr[j] - max);

            reduceProd = logf(reduceProd);
            for (size_t j = 0; j < reducedAxisSize; ++j)
                dstDataPtr[j] = srcDataPtr[j] - max - reduceProd;
        });
    } else {
        parallel_for2d(axisStep, reducedAxisStride, [&](size_t k, size_t i) {
            const float *srcDataPtr = &srcData[k * reducedAxisStride * reducedAxisSize + i];
            float *dstDataPtr = &dstData[k * reducedAxisStride * reducedAxisSize + i];

            float reduceProd = 0.0f;
            float max = std::numeric_limits<float>::min();
            for (size_t j = 0; j < reducedAxisSize; ++j) {
                if (srcDataPtr[j * reducedAxisStride] > max)
                    max = srcDataPtr[j * reducedAxisStride];
            }

            for (size_t j = 0; j < reducedAxisSize; ++j)
                reduceProd += expf(srcDataPtr[j * reducedAxisStride] - max);

            reduceProd = logf(reduceProd);
            for (size_t j = 0; j < reducedAxisSize; ++j)
                dstDataPtr[j * reducedAxisStride] = srcDataPtr[j * reducedAxisStride] - max - reduceProd;
        });
    }
}

bool MKLDNNLogSoftmaxNode::created() const {
    return getType() == LogSoftmax;
}

REG_MKLDNN_PRIM_FOR(MKLDNNLogSoftmaxNode, LogSoftmax)
