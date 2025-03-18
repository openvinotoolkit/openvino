// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_topkrois.h"

#include <algorithm>
#include <openvino/opsets/opset6.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu::node {

bool ExperimentalDetectronTopKROIs::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                         std::string& errorMessage) noexcept {
    try {
        const auto topKROI = ov::as_type_ptr<const ov::opset6::ExperimentalDetectronTopKROIs>(op);
        if (!topKROI) {
            errorMessage = "Only opset6 ExperimentalDetectronTopKROIs operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronTopKROIs::ExperimentalDetectronTopKROIs(const std::shared_ptr<ov::Node>& op,
                                                             const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto topKROI = ov::as_type_ptr<const ov::opset6::ExperimentalDetectronTopKROIs>(op);
    if (topKROI == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of ExperimentalDetectronTopKROIs from opset6.");
    }

    if (inputShapes.size() != 2 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    if (getInputShapeAtPort(INPUT_ROIS).getRank() != 2 || getInputShapeAtPort(INPUT_PROBS).getRank() != 1) {
        THROW_CPU_NODE_ERR("has unsupported input shape");
    }

    max_rois_num_ = topKROI->get_max_rois();
}

void ExperimentalDetectronTopKROIs::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronTopKROIs::execute(const dnnl::stream& strm) {
    const int input_rois_num = getParentEdgeAt(INPUT_ROIS)->getMemory().getStaticDims()[0];
    const int top_rois_num = (std::min)(max_rois_num_, input_rois_num);

    auto* input_rois = getSrcDataAtPortAs<const float>(INPUT_ROIS);
    auto* input_probs = getSrcDataAtPortAs<const float>(INPUT_PROBS);
    auto* output_rois = getDstDataAtPortAs<float>(OUTPUT_ROIS);

    std::vector<size_t> idx(input_rois_num);
    iota(idx.begin(), idx.end(), 0);
    // FIXME. partial_sort is enough here.
    sort(idx.begin(), idx.end(), [&input_probs](size_t i1, size_t i2) {
        return input_probs[i1] > input_probs[i2];
    });

    for (int i = 0; i < top_rois_num; ++i) {
        cpu_memcpy(output_rois + 4 * i, input_rois + 4 * idx[i], 4 * sizeof(float));
    }
}

bool ExperimentalDetectronTopKROIs::created() const {
    return getType() == Type::ExperimentalDetectronTopKROIs;
}

}  // namespace ov::intel_cpu::node
