// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_priorgridgenerator.h"

#include <openvino/opsets/opset6.hpp>
#include <string>

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu::node {

bool ExperimentalDetectronPriorGridGenerator::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                                   std::string& errorMessage) noexcept {
    try {
        const auto priorGridGen = ov::as_type_ptr<const ov::opset6::ExperimentalDetectronPriorGridGenerator>(op);
        if (!priorGridGen) {
            errorMessage = "Only opset6 ExperimentalDetectronPriorGridGenerator operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronPriorGridGenerator::ExperimentalDetectronPriorGridGenerator(const std::shared_ptr<ov::Node>& op,
                                                                                 const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto priorGridGen = ov::as_type_ptr<const ov::opset6::ExperimentalDetectronPriorGridGenerator>(op);
    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    const auto& attr = priorGridGen->get_attrs();
    grid_w_ = attr.w;
    grid_h_ = attr.h;
    stride_h_ = attr.stride_y;
    stride_w_ = attr.stride_x;
}

void ExperimentalDetectronPriorGridGenerator::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, ov::element::f32},
                          {LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronPriorGridGenerator::execute(const dnnl::stream& strm) {
    const int num_priors_ = getParentEdgeAt(INPUT_PRIORS)->getMemory().getStaticDims()[0];
    assert(getParentEdgeAt(INPUT_PRIORS)->getMemory().getStaticDims()[1] == 4);

    // Execute
    const int layer_width = grid_w_ ? grid_w_ : getParentEdgeAt(INPUT_FEATUREMAP)->getMemory().getStaticDims()[3];
    const int layer_height = grid_h_ ? grid_h_ : getParentEdgeAt(INPUT_FEATUREMAP)->getMemory().getStaticDims()[2];
    const float step_w =
        stride_w_ ? stride_w_
                  : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getMemory().getStaticDims()[3]) / layer_width;
    const float step_h =
        stride_h_ ? stride_h_
                  : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getMemory().getStaticDims()[2]) / layer_height;

    const auto* bottom_data_0 = getSrcDataAtPortAs<const float>(0);
    auto* top_data_0 = getDstDataAtPortAs<float>(OUTPUT_ROIS);

    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            for (int s = 0; s < num_priors_; ++s) {
                top_data_0[0] = bottom_data_0[4 * s + 0] + step_w * (w + 0.5f);
                top_data_0[1] = bottom_data_0[4 * s + 1] + step_h * (h + 0.5f);
                top_data_0[2] = bottom_data_0[4 * s + 2] + step_w * (w + 0.5f);
                top_data_0[3] = bottom_data_0[4 * s + 3] + step_h * (h + 0.5f);
                top_data_0 += 4;
            }
        }
    }
}

bool ExperimentalDetectronPriorGridGenerator::created() const {
    return getType() == Type::ExperimentalDetectronPriorGridGenerator;
}

bool ExperimentalDetectronPriorGridGenerator::needPrepareParams() const {
    return false;
}

}  // namespace ov::intel_cpu::node
