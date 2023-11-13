// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include <ngraph/opsets/opset6.hpp>
#include "ie_parallel.hpp"
#include "experimental_detectron_priorgridgenerator.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool ExperimentalDetectronPriorGridGenerator::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                                             std::string& errorMessage) noexcept {
    try {
        const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
        if (!priorGridGen) {
            errorMessage = "Only opset6 ExperimentalDetectronPriorGridGenerator operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronPriorGridGenerator::ExperimentalDetectronPriorGridGenerator(
    const std::shared_ptr<ov::Node>& op,
    const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = "ExperimentalDetectronPriorGridGenerator layer with name '" + op->get_friendly_name() + "'";
    const auto priorGridGen = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronPriorGridGenerator>(op);
    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input/output edges!");

    const auto &attr = priorGridGen->get_attrs();
    grid_w_ = attr.w;
    grid_h_ = attr.h;
    stride_h_ = attr.stride_y;
    stride_w_ = attr.stride_x;
}

void ExperimentalDetectronPriorGridGenerator::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::FP32}},
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronPriorGridGenerator::execute(dnnl::stream strm) {
    const int num_priors_ = getParentEdgeAt(INPUT_PRIORS)->getMemory().getStaticDims()[0];
    assert(getParentEdgeAt(INPUT_PRIORS)->getMemory().getStaticDims()[1] == 4);

    // Execute
    const int layer_width = grid_w_ ? grid_w_ : getParentEdgeAt(INPUT_FEATUREMAP)->getMemory().getStaticDims()[3];
    const int layer_height = grid_h_ ? grid_h_ : getParentEdgeAt(INPUT_FEATUREMAP)->getMemory().getStaticDims()[2];
    const float step_w = stride_w_ ? stride_w_ : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getMemory().getStaticDims()[3]) / layer_width;
    const float step_h = stride_h_ ? stride_h_ : static_cast<float>(getParentEdgeAt(INPUT_IMAGE)->getMemory().getStaticDims()[2]) / layer_height;

    const auto *bottom_data_0 = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    auto *top_data_0 = reinterpret_cast<float *>(getChildEdgesAtPort(OUTPUT_ROIS)[0]->getMemoryPtr()->getData());

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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
