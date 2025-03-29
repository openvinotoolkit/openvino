// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot.h"

#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "dnnl_types.h"
#include "nodes/common/blocked_desc_creator.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "selective_build.h"
#include "shape_inference/custom/one_hot.hpp"

namespace ov::intel_cpu::node {

bool OneHot::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto oneHot = ov::as_type_ptr<const ov::opset1::OneHot>(op);
        if (!oneHot) {
            errorMessage = "Only opset1 OneHot operation is supported";
            return false;
        }
        if (ov::as_type_ptr<const ov::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID)) == nullptr) {
            errorMessage = "Only const 'on_value' input is supported";
            return false;
        }
        if (ov::as_type_ptr<const ov::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID)) ==
            nullptr) {
            errorMessage = "Only const 'off_value' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

OneHot::OneHot(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, OneHotShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto oneHot = ov::as_type_ptr<const ov::opset1::OneHot>(op);
    const auto depthNode = ov::as_type_ptr<const ov::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID));
    if (depthNode) {
        depth = depthNode->cast_vector<uint32_t>()[0];
    }
    axis = oneHot->get_axis();

    VectorDims srcDims = getInputShapeAtPort(INDICES_ID).getDims();
    if (ov::is_scalar(srcDims)) {
        srcDims = VectorDims{1};
    }
    VectorDims dstDims = getOutputShapeAtPort(0).getDims();
    if (ov::is_scalar(dstDims)) {
        dstDims = VectorDims{1};
    }

    int output_dims_size = dstDims.size();
    if (axis < 0) {
        axis += output_dims_size;
    }
    if (axis < 0 || axis >= output_dims_size) {
        THROW_CPU_NODE_ERR("has unsupported 'axis' attribute: ", oneHot->get_axis());
    }

    if (!(((1 + srcDims.size()) == dstDims.size()) ||
          (depthNode && (srcDims.size() == 1 && dstDims.size() == 1 && dstDims[0] == depth && srcDims[0] == 1)))) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output dimensions!");
    }
}

bool OneHot::needShapeInfer() const {
    const auto depthNodePtr = getSrcDataAtPortAs<int32_t>(1);
    if (depth != static_cast<size_t>(depthNodePtr[0])) {
        depth = depthNodePtr[0];
        return true;
    }

    return Node::needShapeInfer();
}

void OneHot::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    // check a precision of the input tensor
    auto input_precision = getOriginalInputPrecisionAtPort(INDICES_ID);
    if (input_precision != ov::element::i32) {
        THROW_CPU_NODE_ERR("has incorrect input precision for the input. Only I32 is supported!");
    }
    output_precision = getOriginalOutputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, input_precision},
                          {LayoutType::ncsp, input_precision},
                          {LayoutType::ncsp, output_precision},
                          {LayoutType::ncsp, output_precision}},
                         {{LayoutType::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

template <typename out_type>
void OneHot::one_hot(size_t prefix_size, size_t suffix_size) {
    const auto* src_data = getSrcDataAtPortAs<const in_type>(0);
    auto* dst_data = getDstDataAtPortAs<out_type>(0);

    const out_type on_value = getSrcDataAtPortAs<const out_type>(2)[0];
    const out_type off_value = getSrcDataAtPortAs<const out_type>(3)[0];

    // fill the output with off_value
    std::size_t dst_size = prefix_size * depth * suffix_size;
    std::fill(dst_data, dst_data + dst_size, off_value);

    // set on_value at needed locations
    auto on_val = on_value;
    parallel_for(prefix_size, [&](std::size_t prefix_idx) {
        const in_type* src_dataPtr = &src_data[prefix_idx * suffix_size];
        out_type* dst_dataPtr = &dst_data[prefix_idx * depth * suffix_size];
        for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; ++suffix_idx, ++src_dataPtr, ++dst_dataPtr) {
            auto v = static_cast<std::size_t>(*src_dataPtr);
            if (v < depth) {
                dst_dataPtr[v * suffix_size] = on_val;
            }
        }
    });
}

void OneHot::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void OneHot::execute(const dnnl::stream& strm) {
    std::size_t prefix_size = 1;
    auto input_dims = getParentEdgeAt(0)->getMemory().getStaticDims();

    std::size_t actual_axis = (axis == -1) ? input_dims.size() : axis;
    for (size_t i = 0; i < actual_axis; ++i) {
        prefix_size *= input_dims[i];
    }

    std::size_t suffix_size = getParentEdgeAt(0)->getMemory().getShape().getElementsCount() / prefix_size;

    OneHotContext ctx = {this, prefix_size, suffix_size};
    OV_SWITCH(intel_cpu,
              OneHotExecute,
              ctx,
              output_precision.size(),
              OV_CASE(sizeof(uint32_t), uint32_t),
              OV_CASE(sizeof(uint16_t), uint16_t),
              OV_CASE(sizeof(uint8_t), uint8_t))
}

bool OneHot::created() const {
    return getType() == Type::OneHot;
}

}  // namespace ov::intel_cpu::node
