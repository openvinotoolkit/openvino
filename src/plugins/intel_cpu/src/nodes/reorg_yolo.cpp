// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo.h"

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

bool ReorgYolo::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto reorgYolo = ov::as_type_ptr<const ov::op::v0::ReorgYolo>(op);
        if (!reorgYolo) {
            errorMessage = "Only v0 ReorgYolo operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ReorgYolo::ReorgYolo(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() != 1 || getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    const auto reorgYolo = ov::as_type_ptr<const ov::op::v0::ReorgYolo>(op);
    const auto strides = reorgYolo->get_strides();
    if (strides.empty()) {
        THROW_CPU_NODE_ERR("has empty strides");
    }
    stride = strides[0];
}

void ReorgYolo::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ReorgYolo::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void ReorgYolo::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto* src_data = getSrcDataAtPortAs<const float>(0);
    auto* dst_data = getDstDataAtPortAs<float>(0);

    const auto& inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    const int IW = (inDims.size() > 3) ? inDims[3] : 1;
    const int IH = (inDims.size() > 2) ? inDims[2] : 1;
    const int IC = (inDims.size() > 1) ? inDims[1] : 1;
    const int B = (!inDims.empty()) ? inDims[0] : 1;

    const int ic_off = IC / (stride * stride);
    const int ih_off = IH * stride;
    const int iw_off = IW * stride;
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int ih = 0; ih < IH; ih++) {
                for (int iw = 0; iw < IW; iw++) {
                    const int dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;

                    const int oc = ic % ic_off;
                    const int offset = ic / ic_off;

                    const int ow = iw * stride + offset % stride;
                    const int oh = ih * stride + offset / stride;

                    const int srcIndex = b * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;

                    dst_data[dstIndex] = src_data[srcIndex];
                }
            }
        }
    }
}

bool ReorgYolo::created() const {
    return getType() == Type::ReorgYolo;
}

}  // namespace ov::intel_cpu::node
