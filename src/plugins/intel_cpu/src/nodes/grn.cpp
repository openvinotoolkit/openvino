// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/opsets/opset1.hpp"
#include "openvino/core/parallel.hpp"
#include "grn.h"

namespace ov {
namespace intel_cpu {
namespace node {

bool GRN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto grn = std::dynamic_pointer_cast<const ov::opset1::GRN>(op);
        if (!grn) {
            errorMessage = "Only opset1 GRN operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GRN::GRN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = "GRN layer with name '" + op->get_friendly_name() + "'";
    const auto grn = std::dynamic_pointer_cast<const ov::opset1::GRN>(op);
    if (grn == nullptr)
        OPENVINO_THROW("Operation with name '", op->get_friendly_name(), "' is not an instance of GRN from opset1.");

    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input/output edges!");

    const auto dataRank = getInputShapeAtPort(0).getRank();

    if (dataRank != getOutputShapeAtPort(0).getRank())
        OPENVINO_THROW(errorPrefix, " has input/output rank mismatch");

    bias = grn->get_bias();
}

void GRN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32, false, 0}},
                         {{LayoutType::ncsp, ov::element::f32, false, 0}},
                         impl_desc_type::ref_any);
}

void GRN::prepareParams() {
    const auto& dataMemPtr = getSrcMemoryAtPort(0);
    const auto& dstMemPtr = getDstMemoryAtPort(0);

    if (!dataMemPtr || !dataMemPtr->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined input memory");
    if (!dstMemPtr || !dstMemPtr->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined output memory");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        OPENVINO_THROW(errorPrefix, " has unidentified preferable primitive descriptor");

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    for (size_t i = 0; i < dataDims.size(); ++i) {
        if (dataDims[i] != dstDims[i])
            OPENVINO_THROW(errorPrefix, " hsd input/output tensors dimensions mismatch");
    }

    if (dataDims.size() > 0)
        N = static_cast<int>(dataDims[0]);
    if (dataDims.size() > 1)
        C = static_cast<int>(dataDims[1]);
    if (dataDims.size() > 2)
        H = static_cast<int>(dataDims[2]);
    if (dataDims.size() > 3)
        W = static_cast<int>(dataDims[3]);
}

void GRN::executeDynamicImpl(dnnl::stream strm) {
    execute(std::move(strm));
}

void GRN::execute(dnnl::stream strm) {
    const float* src_data = getSrcDataAtPortAs<const float>(0);
    float* dst_data = getDstDataAtPortAs<float>(0);

    parallel_for3d(N, H, W, [&](int b, int h, int w) {
        double variance = 0;
        for (int c = 0; c < C; c++) {
            variance += std::pow(src_data[b*C*H*W + c*H*W + h*W + w], 2);
        }
        variance = std::pow(variance + bias, 0.5f);
        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] / static_cast<float>(variance);
        }
    });
}

bool GRN::created() const {
    return getType() == Type::GRN;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
