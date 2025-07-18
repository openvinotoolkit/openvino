// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_topkrois.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/experimental_detectron_topkrois.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

bool ExperimentalDetectronTopKROIs::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                         std::string& errorMessage) noexcept {
    try {
        const auto topKROI = ov::as_type_ptr<const ov::op::v6::ExperimentalDetectronTopKROIs>(op);
        if (!topKROI) {
            errorMessage = "Only v6 ExperimentalDetectronTopKROIs operation is supported";
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

    const auto topKROI = ov::as_type_ptr<const ov::op::v6::ExperimentalDetectronTopKROIs>(op);
    CPU_NODE_ASSERT(topKROI, "is not an instance of ExperimentalDetectronTopKROIs from opset6.");

    CPU_NODE_ASSERT(inputShapes.size() == 2 && outputShapes.size() == 1, "has incorrect number of input/output edges!");

    CPU_NODE_ASSERT(getInputShapeAtPort(INPUT_ROIS).getRank() == 2 && getInputShapeAtPort(INPUT_PROBS).getRank() == 1,
                    "has unsupported input shape");

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

void ExperimentalDetectronTopKROIs::execute([[maybe_unused]] const dnnl::stream& strm) {
    const int input_rois_num = getParentEdgeAt(INPUT_ROIS)->getMemory().getStaticDims()[0];
    const int top_rois_num = (std::min)(max_rois_num_, input_rois_num);

    const auto* input_rois = getSrcDataAtPortAs<const float>(INPUT_ROIS);
    const auto* input_probs = getSrcDataAtPortAs<const float>(INPUT_PROBS);
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
