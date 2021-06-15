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
#include "mkldnn_broadcast_node.h"
#include <nodes/common/tensor_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNBroadcastNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto broadcast = std::dynamic_pointer_cast<const ngraph::opset1::Broadcast>(op);
        if (!broadcast) {
            errorMessage = "Only opset1 Broadcast operation is supported";
            return false;
        }
        if (broadcast->get_broadcast_spec() != ngraph::op::AutoBroadcastSpec::NUMPY) {
            errorMessage = "Only NUMPY broadcast type is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(broadcast->get_input_node_shared_ptr(BROADCAST_SHAPE)) == nullptr) {
            errorMessage = "Only const 'shape' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNBroadcastNode::MKLDNNBroadcastNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Broadcast node with name '" + op->get_friendly_name() + "'";
    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    SizeVector shape_dims = op->get_input_shape(BROADCAST_SHAPE);
    if (shape_dims.size() > 1)
        IE_THROW() << errorPrefix << " has incorrect 'shape' input rank: " << shape_dims.size();
}

void MKLDNNBroadcastNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision prec = getOriginalInputPrecisionAtPort(BROADCAST_INPUT);

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, prec},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         {{TensorDescCreatorTypes::ncsp, prec}},
                         impl_desc_type::ref_any);
}

void MKLDNNBroadcastNode::execute(mkldnn::stream strm) {
    size_t shape_size = (getParentEdgeAt(BROADCAST_SHAPE)->getDesc().getDims())[0];
    SizeVector dst_dims = getChildEdgeAt(0)->getDesc().getDims();
    SizeVector src_dims = getParentEdgeAt(BROADCAST_INPUT)->getDesc().getDims();
    SizeVector srcStrides = getParentEdgeAt(BROADCAST_INPUT)->getDesc().getBlockingDesc().getStrides();
    size_t data_size = getParentEdgeAt(BROADCAST_INPUT)->getDesc().getPrecision().size();

    if (!src_dims.size())
        src_dims = SizeVector(1, 1);
    if (!srcStrides.size())
        srcStrides = SizeVector(1, 1);

    if (dst_dims.size() != shape_size) {
        IE_THROW() << "Output tensor dimension mismatch";
    }

    if (src_dims.size() > dst_dims.size()) {
        IE_THROW() << "Output tensor dimension is smaller then input tensor dimension";
    }

    InferenceEngine::SizeVector dstStrides = getChildEdgeAt(0)->getDesc().getBlockingDesc().getStrides();
    InferenceEngine::SizeVector src_aligned(dst_dims.size());
    InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
    size_t prefix_size = dst_dims.size() - src_dims.size();
    for (size_t i = 0; i < dst_dims.size(); i++) {
        if (i < prefix_size) {
            src_aligned[i] = 1;
            srcStrides_aligned[i] = srcStrides[0];
        } else {
            src_aligned[i] = src_dims[i - prefix_size];
            srcStrides_aligned[i] = srcStrides[i - prefix_size];
        }
    }

    size_t work_amount_dst = dstStrides[0] * dst_dims[0];
    const auto *src_data = reinterpret_cast<const uint8_t *>(getParentEdgeAt(BROADCAST_INPUT)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i, src_idx, start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (int j = dst_dims.size() - 1, i = start; j >= 0; j--) {
            counters[j] = i % dst_dims[j];
            i /= dst_dims[j];
        }
        for (size_t iwork = start * data_size; iwork < end * data_size; iwork += data_size) {
            for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
                src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

            cpu_memcpy(&dst_data[iwork], &src_data[src_idx * data_size], data_size);

            for (int j = dst_dims.size() - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % dst_dims[j];
                if (counters[j] != 0) break;
            }
        }
    });
}

bool MKLDNNBroadcastNode::created() const {
    return getType() == Broadcast;
}

REG_MKLDNN_PRIM_FOR(MKLDNNBroadcastNode, Broadcast)
