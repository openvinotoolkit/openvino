// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "../mkldnn_extension_utils.h"
#include <string>
#include <tuple>
#include <algorithm>
#include "caseless.hpp"
#include "ie_memcpy.h"

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
    memory::format outFormat = mkldnn::memory::format_undef;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getCnnLayer()->outData[0]->getPrecision();
        if (precision == InferenceEngine::Precision::U16 || isMeanImage) {
            precision = InferenceEngine::Precision::FP32;
        }
        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        outFormat = MKLDNNMemory::Convert(getCnnLayer()->outData[0]->getLayout());
        dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, outFormat);
        config.outConfs.push_back(dataConfig);
    } else if (getType() == Output) {
        precision = getCnnLayer()->insData[0].lock()->getPrecision();
        if (precision == InferenceEngine::Precision::U16) precision = InferenceEngine::Precision::FP32;
        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        outFormat = MKLDNNMemory::Convert(getCnnLayer()->insData[0].lock()->getLayout());
        dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, outFormat);
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormat);
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

    bool isCompatibleTensors(const InferenceEngine::TensorDesc &lhs, const InferenceEngine::TensorDesc &rhs) {
        auto const &lhsBlockingDesc = lhs.getBlockingDesc();
        auto const &rhsBlockingDesc = rhs.getBlockingDesc();

        bool lhsDefaultStrides, rhsDefaultStrides;
        size_t lhsSize, rhsSize;

        std::tie(lhsDefaultStrides, lhsSize) = isDefaultStrides(lhsBlockingDesc.getStrides(), lhs.getDims());
        std::tie(rhsDefaultStrides, rhsSize) = isDefaultStrides(rhsBlockingDesc.getStrides(), rhs.getDims());

        return lhs.getPrecision() == rhs.getPrecision()
                && lhsSize == rhsSize
                && lhsDefaultStrides
                && rhsDefaultStrides
                && isDefaultOrder(lhsBlockingDesc.getOrder())
                && isDefaultOrder(rhsBlockingDesc.getOrder());
    }
}   // namespace

void MKLDNNInputNode::execute(mkldnn::stream strm) {
    if (!constBlob)
        return;
    auto dstBlob = getChildEdgeAt(0)->getBlob();

    if (constBlob->size() != dstBlob->size()) {
        THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
    }

    if (constBlob->getTensorDesc() == dstBlob->getTensorDesc()
        || isCompatibleTensors(constBlob->getTensorDesc(), dstBlob->getTensorDesc())) {
        const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
        int8_t *dstData = dstBlob->buffer();

        ie_memcpy(dstData, dstBlob->byteSize(), srcData, constBlob->byteSize());
    } else {
        switch (precision.size()) {
            case 1: {
                const int8_t *srcData = constBlob->cbuffer().as<int8_t *>();
                int8_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 2: {
                const int16_t *srcData = constBlob->cbuffer().as<int16_t *>();
                int16_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 4: {
                const int32_t *srcData = constBlob->cbuffer().as<int32_t *>();
                int32_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            case 8: {
                const int64_t *srcData = constBlob->cbuffer().as<int64_t *>();
                int64_t *dstData = dstBlob->buffer();

                for (size_t i = 0; i < constBlob->size(); i++)
                    dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];

                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision for node " << getName();
        }
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
