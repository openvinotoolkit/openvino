// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_transpose_node.h"

#include <algorithm>
#include <string>
#include <mkldnn_extension_utils.h>
#include <mkldnn_selective_build.h>
#include <cpu/x64/jit_generator.hpp>
#include "common/tensor_desc_creator.h"
#include <utils/general_utils.h>
#include "utils/bfloat16.hpp"


using namespace mkldnn;
using namespace mkldnn::impl::cpu::x64;
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

    auto dataPrecision = getOriginalInputPrecisionAtPort(0);
    auto orderPrecision = getOriginalInputPrecisionAtPort(1);

    impl_desc_type impl_type;
    if (mayiuse(impl::cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(impl::cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(impl::cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    auto canUseBlocked = [=](const size_t block) {
        return getParentEdgeAt(0)->getDims()[1] % block == 0;
    };

    DataConfigurator orderDC(TensorDescCreatorTypes::ncsp, orderPrecision);
    addSupportedPrimDesc({{ TensorDescCreatorTypes::ncsp, dataPrecision}, orderDC},
                         {{ TensorDescCreatorTypes::ncsp, dataPrecision}},
                         impl_type, true);

    if (canUseBlocked(16))
        addSupportedPrimDesc({{TensorDescCreatorTypes::nCsp16c, dataPrecision}, orderDC},
                             {{TensorDescCreatorTypes::ncsp, dataPrecision}},
                             impl_type, true);
    if (canUseBlocked(8))
        addSupportedPrimDesc({{ TensorDescCreatorTypes::nCsp8c, dataPrecision}, orderDC},
                             {{ TensorDescCreatorTypes::ncsp, dataPrecision}},
                             impl_type, true);
    if (one_of(dataPrecision, Precision::FP32, Precision::I8, Precision::U8))
        addSupportedPrimDesc({{TensorDescCreatorTypes::nspc, dataPrecision}, orderDC},
                             {{TensorDescCreatorTypes::nspc, dataPrecision}},
                             impl_type, true);
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
