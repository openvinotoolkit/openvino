// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "../mkldnn_extension_utils.h"
#include <string>
#include <tuple>
#include <algorithm>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

MKLDNNInputNode::MKLDNNInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    constant = ConstantType::NoConst;
    if (layer && CaselessEq<std::string>()(layer->type, "const")) {
        constant = ConstantType::Const;
        if (layer->blobs.size() != 1 || getType() != Input || !layer->blobs.begin()->second)
            THROW_IE_EXCEPTION << "Incorrect const input " << getName();
        constBlob = layer->blobs.begin()->second;
    } else {
        constBlob = nullptr;
    }
}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getCnnLayer()->outData[0]->getPrecision();
        if (precision == InferenceEngine::Precision::U16 || isMeanImage) {
            precision = InferenceEngine::Precision::FP32;
        }
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto mem_tdesc = MKLDNNMemoryDesc(getCnnLayer()->outData[0]->getTensorDesc());
        dataConfig.desc = mem_tdesc;
        config.outConfs.push_back(dataConfig);
    } else if (getType() == Output) {
        precision = getCnnLayer()->insData[0].lock()->getPrecision();
        if (precision == InferenceEngine::Precision::U16) precision = InferenceEngine::Precision::FP32;
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto mem_tdesc = MKLDNNMemoryDesc(getCnnLayer()->insData[0].lock()->getTensorDesc());
        dataConfig.desc = mem_tdesc;
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

namespace {
    bool isDefaultOrder(const InferenceEngine::SizeVector &order) {
        return std::is_sorted(order.begin(), order.end(),
                                [](size_t a, size_t b) { return a + 1 == b; });
    }

    std::tuple<bool, size_t> isDefaultStrides(const InferenceEngine::SizeVector &strides,
                                              const InferenceEngine::SizeVector &dims) {
        if (strides.size() != dims.size())
            return std::make_tuple(false, 0);

        size_t dim = 1;

        for (size_t i = dims.size(); i-- > 0;) {
            if (strides[i] != dim)
                return std::make_tuple(false, 0);
            dim *= dims[i];
        }

        return std::make_tuple(true, dim);
    }

    bool isCompatibleTensors(const InferenceEngine::TensorDesc &lhs, const InferenceEngine::TensorDesc &rhs,
                             bool isNeedPrecValid = true) {
        auto const &lhsBlockingDesc = lhs.getBlockingDesc();
        auto const &rhsBlockingDesc = rhs.getBlockingDesc();

        bool lhsDefaultStrides = false, rhsDefaultStrides = false;
        size_t lhsSize = 0lu, rhsSize = 0lu;

        std::tie(lhsDefaultStrides, lhsSize) = isDefaultStrides(lhsBlockingDesc.getStrides(), lhs.getDims());
        std::tie(rhsDefaultStrides, rhsSize) = isDefaultStrides(rhsBlockingDesc.getStrides(), rhs.getDims());
        bool isCompatTensors = lhsSize == rhsSize
                               && lhsDefaultStrides
                               && rhsDefaultStrides
                               && isDefaultOrder(lhsBlockingDesc.getOrder())
                               && isDefaultOrder(rhsBlockingDesc.getOrder());

        return (isNeedPrecValid ? lhs.getPrecision() == rhs.getPrecision() : true) && isCompatTensors;
    }
}   // namespace

void MKLDNNInputNode::execute(mkldnn::stream strm) {
    if (!constBlob)
        return;
    auto dstBlob = getChildEdgeAt(0)->getBlob();

    if (constBlob->getTensorDesc() == dstBlob->getTensorDesc()
        || isCompatibleTensors(constBlob->getTensorDesc(), dstBlob->getTensorDesc())) {
        const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
        int8_t *dstData = dstBlob->buffer();

        cpu_memcpy_s(dstData, dstBlob->byteSize(), srcData, constBlob->byteSize());
    } else if (constBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::BIN ||
               dstBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::BIN) {
        size_t dstSize = dstBlob->size() / 8;
        if (constBlob->size() != dstSize) {
            THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
        }

        const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
        int8_t *dstData = dstBlob->buffer();

        cpu_memcpy_s(dstData, dstSize, srcData, constBlob->byteSize());
    } else if (constBlob->getTensorDesc().getPrecision() != dstBlob->getTensorDesc().getPrecision() &&
               isCompatibleTensors(constBlob->getTensorDesc(), dstBlob->getTensorDesc(), false)) {
        cpu_convert(constBlob->cbuffer().as<const void *>(), dstBlob->buffer().as<void *>(),
                    constBlob->getTensorDesc().getPrecision(), dstBlob->getTensorDesc().getPrecision(), dstBlob->size());
    } else {
        THROW_IE_EXCEPTION << "Input node with name: '" << getName() << "' has incompatible tensors";
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Output);
