// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_transpose_node.h"

#include <algorithm>
#include <string>
#include <mkldnn_extension_utils.h>
#include <mkldnn_selective_build.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"


using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;


bool MKLDNNTransposeNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto transposeOp = ngraph::as_type_ptr<const ngraph::op::v1::Transpose>(op);
        if (!transposeOp) {
            errorMessage = "Node is not an instance of the Transpose operation.";
            return false;
        }

        auto orderOp = ngraph::as_type_ptr<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!orderOp) {
            errorMessage = "Constant expected as the second input.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTransposeNode::MKLDNNTransposeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto orderOp = ngraph::as_type_ptr<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    order = orderOp->cast_vector<size_t>();

    if (order.empty()) {
        size_t rank = op->get_input_shape(0).size();
        for (size_t i = 1lu; i <= rank; ++i) {
            order.emplace_back(rank - i);
        }
    }
}

void MKLDNNTransposeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    prec = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
    auto inputOrderDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(1));

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), inputOrderDataType, memory::format_tag::x);
    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nchw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nchw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nchw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nChw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nChw16c});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nhwc);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nhwc);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nhwc});
        }
    } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ncdhw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ncdhw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::ncdhw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nCdhw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nCdhw16c});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ndhwc);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ndhwc);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::ndhwc});
        }
    } else {
        // general plain case
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    }
}

void MKLDNNTransposeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    PermuteParams params;
    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    params.order = order;

    auto srcDesc = getParentEdgeAt(0)->getDesc();
    params.src_block_dims = srcDesc.getBlockingDesc().getBlockDims();
    params.src_block_order = srcDesc.getBlockingDesc().getOrder();

    auto dstDesc = getChildEdgeAt(0)->getDesc();
    params.dst_block_dims = dstDesc.getBlockingDesc().getBlockDims();
    params.dst_block_order = dstDesc.getBlockingDesc().getOrder();

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNTransposeNode::execute(mkldnn::stream strm) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    permuteKernel->execute(srcData, dstData, batchToProcess());
}

bool MKLDNNTransposeNode::created() const {
    return getType() == Transpose;
}
REG_MKLDNN_PRIM_FOR(MKLDNNTransposeNode, Transpose);
