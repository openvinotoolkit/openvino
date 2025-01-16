// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile.h"

#include "openvino/op/tile.hpp"
#include "openvino/op/constant.hpp"

#include "common/cpu_memcpy.h"
#include "utils/ngraph_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool Tile::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v0::Tile>(op)) {
            errorMessage = "Only opset1 Tile operation is supported.";
            return false;
        }
        if (op->get_input_partial_shape(TILE_REPEATS).is_dynamic()) {
            errorMessage = "Only static shape is supported for tile repeats input.";
            return false;
        }
        if (!isDynamicNgraphNode(op) &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(TILE_REPEATS))) {
            errorMessage = "Only constant 'Repeats' input is supported with static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Tile::Tile(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, NgraphShapeInferFactory(op, PortMask(TILE_REPEATS))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = "Tile node with name '" + getName() + "'";

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(TILE_REPEATS))) {
        constMap[TILE_REPEATS] = true;
        repeats = originRepeats = ov::as_type<const ov::op::v0::Constant>(op->get_input_node_ptr(TILE_REPEATS))->cast_vector<size_t>();
        while (repeats.size() < getInputShapeAtPort(TILE_INPUT).getRank()) {
            repeats.insert(repeats.begin(), 1lu);
        }
    }
}

void Tile::getSupportedDescriptors() {
    const auto& vec_to_string = [](const std::vector<size_t>& vec) -> std::string {
        std::string result = "[";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i)
                result += ", ";
            result += std::to_string(vec[i]);
        }
        return result;
    };
    if (getParentEdges().size() != 2)
        OPENVINO_THROW(errorPrefix,
                       " has incorrect number of input edges. "
                       "Expected: 2, Actual: ",
                       getParentEdges().size());
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, " has no output edges.");
    const auto& dstDims0 = getOutputShapeAtPort(0).getDims();
    for (size_t i = 1lu; i < outputShapes.size(); i++) {
        const auto& dstDims = getOutputShapeAtPort(i).getDims();
        if (dstDims.size() != dstDims0.size())
            OPENVINO_THROW(errorPrefix,
                           " has output edges 0 and ",
                           i,
                           " with different ranks: ",
                           dstDims0.size(),
                           " and ",
                           dstDims.size());
        for (size_t j = 0; j < dstDims0.size(); j++) {
            if (dstDims0[j] != dstDims[j]) {
                OPENVINO_THROW(errorPrefix,
                               " has output edges 0 and ",
                               i,
                               " with different dims: ",
                               vec_to_string(dstDims0),
                               " and ",
                               vec_to_string(dstDims));
            }
        }
    }
    if (constMap[TILE_REPEATS] && getInputShapeAtPort(TILE_INPUT).getRank() > getOutputShapeAtPort(0).getRank())
        OPENVINO_THROW(
            errorPrefix,
            " has incorrect input/output data shape rank. Input shape rank cannot be more than output shape rank. "
            "Actual input shape size: ",
            getInputShapeAtPort(TILE_INPUT).getRank(),
            ", output shape size: ",
            getOutputShapeAtPort(0).getRank());

    if (!isDynamicNode())
        needPrepareParamsVar = true;
}

void Tile::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors = getSupportedConfigs(this, outputShapes.size());
}

bool Tile::needPrepareParams() const {
    return needPrepareParamsVar;
}

void Tile::prepareParams() {
    if (!constMap[TILE_REPEATS]) {
        const auto& repeatsMem = getParentEdgeAt(TILE_REPEATS)->getMemory();

        const int32_t* repeatsData = repeatsMem.getDataAs<const int32_t>();
        originRepeats.assign(repeatsData, repeatsData + repeatsMem.getStaticDims()[0]);

        repeats.assign(std::max(originRepeats.size(), getInputShapeAtPort(TILE_INPUT).getRank()), 1lu);
        const size_t offset = repeats.size() - originRepeats.size();
        for (size_t i = 0lu; i < originRepeats.size(); i++) {
            repeats[i + offset] = originRepeats[i];
        }
    }

    auto srcBlockedDims = getParentEdgeAt(TILE_INPUT)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto dstBlockedDims = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

bool Tile::needShapeInfer() const {
    needPrepareParamsVar = true;
    if (inputShapesModified()) {
        return true;
    }
    if (!constMap[TILE_REPEATS]) {
        if (originRepeats.empty())
            return true;
        const int32_t* repeatsData = getSrcDataAtPortAs<const int32_t>(TILE_REPEATS);
        for (size_t i = 0lu; i < originRepeats.size(); i++) {
            if (originRepeats[i] != static_cast<size_t>(repeatsData[i]))
                return true;
        }
    }
    needPrepareParamsVar = false;
    return false;
}

void Tile::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Tile::execute(dnnl::stream strm) {
    if (optimizedCase) {
        optimizedExecute(getSrcMemoryAtPort(TILE_INPUT), getDstMemoryAtPort(0));
    } else {
        plainExecute(strm);
    }
}

void Tile::plainExecute(dnnl::stream strm) {
    if (noTiling) {
        return;
    }

    auto& srcMemory = getParentEdgeAt(TILE_INPUT)->getMemory();

    const uint8_t* src_ptr = srcMemory.getDataAs<const uint8_t>();
    uint8_t* dst_ptr = getDstDataAtPortAs<uint8_t>(0);

    int m_inner_dim = 1;
    int m_outer_dim = 1;
    auto inDims = srcMemory.getStaticDims();
    for (int i = 0; i < axis; i++ )
        m_outer_dim *= inDims[i];
    for (size_t i = axis; i < inDims.size(); i++ )
        m_inner_dim *= inDims[i];

    int MB = srcMemory.getStaticDims()[0];
    if (axis > 0) {
        m_outer_dim /= inDims[0];
        m_outer_dim *= MB;
    } else {
        m_inner_dim /= inDims[0];
        m_inner_dim *= MB;
    }

    if (m_inner_dim == 1 && m_outer_dim % 8 == 0 && srcMemory.getDesc().hasLayoutType(LayoutType::nCsp8c)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw8c)
         */
        m_inner_dim *= 8;
        m_outer_dim /= 8;
    } else if (m_inner_dim == 1 && m_outer_dim % 16 == 0 && srcMemory.getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw16c)
         */
        m_inner_dim *= 16;
        m_outer_dim /= 16;
    }

    m_inner_dim *= srcMemory.getDesc().getPrecision().size();
    for (int i = 0; i < m_outer_dim; ++i) {
        for (int t = 0; t < tiles; ++t) {
            cpu_memcpy(dst_ptr, src_ptr, m_inner_dim);
            dst_ptr += m_inner_dim;
        }
        src_ptr += m_inner_dim;
    }
}

bool Tile::created() const {
    return getType() == Type::Tile;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
