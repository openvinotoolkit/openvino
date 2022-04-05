// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "grn.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool GRN::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);
        if (!grn) {
            errorMessage = "Only opset1 GRN operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GRN::GRN(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "GRN layer with name '" + op->get_friendly_name() + "'";
    const auto grn = std::dynamic_pointer_cast<const ngraph::opset1::GRN>(op);
    if (grn == nullptr)
        IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
            "' is not an instance of GRN from opset1.";

    if (getOriginalInputsNumber() != 1 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    bias = grn->get_bias();
}

void GRN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32, false, 0}},
                         {{LayoutType::ncsp, Precision::FP32, false, 0}},
                         impl_desc_type::ref_any);
}

void GRN::execute(dnnl::stream strm) {
    const float* src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    float* dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    const auto &dims = getParentEdgeAt(0)->getMemory().getStaticDims();

    int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
    int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
    int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
    int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

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
