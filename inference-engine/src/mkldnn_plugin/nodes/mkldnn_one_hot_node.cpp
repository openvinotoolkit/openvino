// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include "mkldnn_one_hot_node.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNOneHotNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
        if (!oneHot) {
            errorMessage = "Only opset1 OneHot operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID)) == nullptr) {
            errorMessage = "Only const 'depth' input is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID)) == nullptr) {
            errorMessage = "Only const 'on_value' input is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID)) == nullptr) {
            errorMessage = "Only const 'off_value' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNOneHotNode::MKLDNNOneHotNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "OneHot layer with name '" + op->get_friendly_name() + "'";
    const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
    const auto depthNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID));
    const auto onValueNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID));
    const auto offValueNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID));
    depth = depthNode->cast_vector<uint32_t>()[0];
    axis = oneHot->get_axis();
    src_dims = oneHot->get_input_shape(INDICES_ID);
    if (ngraph::is_scalar(src_dims)) {
        src_dims = SizeVector{1};
    }
    dst_dims = oneHot->get_output_shape(0);
    if (ngraph::is_scalar(dst_dims)) {
        dst_dims = SizeVector{1};
    }

    int output_dims_size = dst_dims.size();
    if (axis < 0) {
        axis += output_dims_size;
    }
    if (axis < 0 || axis >= output_dims_size) {
        IE_THROW() << errorPrefix << " has unsupported 'axis' attribute: " << oneHot->get_axis();
    }

    if (!( ((1 + src_dims.size()) == dst_dims.size()) ||
           (src_dims.size() == 1 && dst_dims.size() == 1 && dst_dims[0] == depth && src_dims[0] == 1)))
        IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions!";
}

void MKLDNNOneHotNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // check a precision of the input tensor
    auto input_precision = getOriginalInputPrecisionAtPort(INDICES_ID);
    if (input_precision != Precision::I32) {
        IE_THROW() << errorPrefix << " has incorrect input precision for the input. Only I32 is supported!";
    }
    output_precision = getOriginalOutputPrecisionAtPort(0);

    addSupportedPrimDesc({{GeneralLayout::ncsp, input_precision},
                          {GeneralLayout::ncsp, input_precision},
                          {GeneralLayout::ncsp, output_precision},
                          {GeneralLayout::ncsp, output_precision}},
                         {{GeneralLayout::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

template<typename out_type>
void MKLDNNOneHotNode::one_hot(size_t prefix_size, size_t suffix_size) {
    const auto *src_data = reinterpret_cast<const in_type *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<out_type *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const out_type on_value = reinterpret_cast<const out_type *>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr())[0];
    const out_type off_value = reinterpret_cast<const out_type *>(getParentEdgeAt(3)->getMemoryPtr()->GetPtr())[0];

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

void MKLDNNOneHotNode::execute(mkldnn::stream strm) {
    std::size_t prefix_size = 1;
    auto input_dims = getParentEdgeAt(0)->getShape().getStaticDims();

    std::size_t actual_axis = (axis == -1) ? src_dims.size() : axis;
    for (size_t i = 0; i < actual_axis; ++i)
        prefix_size *= input_dims[i];

    std::size_t suffix_size = getParentEdgeAt(0)->getShape().getElementsCount() / prefix_size;

    OneHotContext ctx = {this, prefix_size, suffix_size};
    OV_SWITCH(MKLDNNPlugin, OneHotExecute, ctx, output_precision.size(),
              OV_CASE(sizeof(uint32_t), uint32_t),
              OV_CASE(sizeof(uint16_t), uint16_t),
              OV_CASE(sizeof(uint8_t), uint8_t))
}

bool MKLDNNOneHotNode::created() const {
    return getType() == OneHot;
}

REG_MKLDNN_PRIM_FOR(MKLDNNOneHotNode, OneHot)
