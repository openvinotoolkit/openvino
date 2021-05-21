// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tile_node.h"
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNTileNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto tile = std::dynamic_pointer_cast<const ngraph::opset1::Tile>(op);
        if (!tile) {
            errorMessage = "Only opset1 Tile operation is supported";
            return false;
        }
        if (tile->get_input_shape(TILE_INPUT).size() != tile->get_input_shape(TILE_REPEATS)[0]) {
            errorMessage = "Doesn't support inputs with different ranks";
            return false;
        }
        const auto repeatsNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(tile->get_input_node_shared_ptr(TILE_REPEATS));
        if (repeatsNode == nullptr) {
            errorMessage = "Only const 'repeats' input is supported";
            return false;
        }
        const auto repeats = repeatsNode->cast_vector<int64_t>();
        if (std::count_if(repeats.begin(), repeats.end(), [](int64_t x) { return x > 1; }) > 1) {
            errorMessage = "Doesn't support 'repeats' with more than one specified axis";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTileNode::MKLDNNTileNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Tile node with name '" + getName() + "'";

        const auto tile = std::dynamic_pointer_cast<const ngraph::opset1::Tile>(op);
        const auto repeatsNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(tile->get_input_node_shared_ptr(TILE_REPEATS));
        const auto repeats = repeatsNode->cast_vector<int64_t>();
        // At this moment CPU plug-in supports tiling only per single axis
        // This behavoiur is guaranteed by ConvertTileToSeqTiles
        for (size_t i = 0; i < repeats.size(); i++) {
            if (repeats[i] > 1) {
                axis = i;
                tiles = repeats[i];
                break;
            }
        }
        noTiling = axis == -1;
        if (axis >= static_cast<int>(tile->get_input_shape(TILE_INPUT).size()))
            IE_THROW() << errorPrefix << " has incorrect tiling axis: " << axis;
        if (tiles < 1 && !noTiling)
            IE_THROW() << errorPrefix << " has incorrect 'repeats' value: " << tiles;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNTileNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (!getChildEdges().size())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
}

void MKLDNNTileNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(TILE_INPUT);
    if (precision.size() != sizeof(PrecisionTrait<Precision::I32>::value_type) &&
        precision.size() != sizeof(PrecisionTrait<Precision::I16>::value_type) &&
        precision.size() != sizeof(PrecisionTrait<Precision::I8>::value_type)) {
        IE_THROW() << errorPrefix << " has unsupported input precision: " << precision;
    }
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto& inDims = getParentEdgeAt(0)->getDims();
    memory::format_tag fmt = MKLDNNMemory::GetPlainFormat(inDims);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[TILE_INPUT].desc = MKLDNNMemoryDesc(getParentEdgeAt(TILE_INPUT)->getDims(), inputDataType, fmt);
    config.inConfs[TILE_REPEATS].desc = MKLDNNMemoryDesc(getParentEdgeAt(TILE_REPEATS)->getDims(), memory::data_type::s32, memory::format_tag::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), inputDataType, fmt);
    config.outConfs[0].inPlace = noTiling ? 0 : -1;
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, fmt});
}

void MKLDNNTileNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " can't get destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " can't get input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";
}

void MKLDNNTileNode::execute(mkldnn::stream strm) {
    if (noTiling) {
        return;
    }

    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(srcMemory.GetPtr());
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetPtr());

    int m_inner_dim = 1;
    int m_outer_dim = 1;
    memory::dims inDims = srcMemory.GetDims();
    for (int i=0; i < axis; i++ ) m_outer_dim *= inDims[i];
    for (int i=axis; i < inDims.size(); i++ ) m_inner_dim *= inDims[i];
    if (axis > 0) {
        m_outer_dim /= inDims[0];
        m_outer_dim *= batchToProcess();
    } else {
        m_inner_dim /= inDims[0];
        m_inner_dim *= batchToProcess();
    }

    if (m_inner_dim == 1 && m_outer_dim % 8 == 0 && srcMemory.GetDesc().isBlockedCFormat(8)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw8c)
         */
        m_inner_dim *= 8;
        m_outer_dim /= 8;
    } else if (m_inner_dim == 1 && m_outer_dim % 16 == 0 && srcMemory.GetDesc().isBlockedCFormat(16)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw16c)
         */
        m_inner_dim *= 16;
        m_outer_dim /= 16;
    }

    m_inner_dim *= srcMemory.GetDesc().GetElementSize();
    for (int i = 0; i < m_outer_dim; ++i) {
        for (int t = 0; t < tiles; ++t) {
            cpu_memcpy(dst_ptr, src_ptr, m_inner_dim);
            dst_ptr += m_inner_dim;
        }
        src_ptr += m_inner_dim;
    }
}

bool MKLDNNTileNode::created() const {
    return getType() == Tile;
}

REG_MKLDNN_PRIM_FOR(MKLDNNTileNode, Tile);
