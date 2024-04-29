// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_broadcast_utils.h"

#include "cpu_convert.h"
#include "cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"



namespace ov {
namespace intel_cpu {

VectorDims TileBroadcastCommon::calculateDenseStrides(const VectorDims &dims) {
    VectorDims strides(dims.size(), 1);

    for (int i = strides.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    return strides;
}

void TileBroadcastCommon::fillOptimizedDimsAndSrcStrides(const VectorDims& srcBlockedDims, const VectorDims& blockedRepeats,
        VectorDims& optimizedDims, VectorDims& optimizedSrcStrides) {
    optimizedDims.clear();
    optimizedSrcStrides.clear();
    VectorDims srcBlockedStrides = calculateDenseStrides(srcBlockedDims);

    for (size_t i = 0; i < srcBlockedDims.size(); i++) {
        optimizedDims.push_back(blockedRepeats[i]);
        optimizedDims.push_back(srcBlockedDims[i]);
        optimizedSrcStrides.push_back(0);
        optimizedSrcStrides.push_back(srcBlockedStrides[i]);
    }

    size_t i = 1;
    while (i < optimizedDims.size() - 1) {
        if (optimizedDims[i] == 1) {
            optimizedDims[i + 1] *= optimizedDims[i - 1];
            optimizedDims.erase(optimizedDims.begin() + i - 1, optimizedDims.begin() + i + 1);
            optimizedSrcStrides.erase(optimizedSrcStrides.begin() + i - 1, optimizedSrcStrides.begin() + i + 1);
        } else {
            i++;
        }
    }

    if (optimizedDims[0] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.begin());
        optimizedSrcStrides.erase(optimizedSrcStrides.begin());
    }

    if (optimizedDims[optimizedDims.size() - 1] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.end() - 1);
        optimizedSrcStrides.erase(optimizedSrcStrides.end() - 1);
    }
}

bool TileBroadcastCommon::canBeExecutedInBlockedLayout(VectorDims srcBlockedDims, VectorDims blockedRepeats,
        const size_t elemsInBlock) {
    if (srcBlockedDims.empty() || blockedRepeats.empty() || elemsInBlock == 0lu || srcBlockedDims[1] == Shape::UNDEFINED_DIM ||
            (blockedRepeats[1] != 1 && srcBlockedDims[1] % elemsInBlock != 0))
        return false;

    srcBlockedDims[1] = div_up(srcBlockedDims[1], elemsInBlock);
    srcBlockedDims.push_back(elemsInBlock);
    blockedRepeats.push_back(1);

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    return optimizedDims.size() <= maxNDims;
}

bool TileBroadcastCommon::canBeExecutedInNSPCLayout(VectorDims srcBlockedDims, VectorDims blockedRepeats) {
    srcBlockedDims.push_back(srcBlockedDims[1]);
    srcBlockedDims.erase(srcBlockedDims.begin() + 1);
    blockedRepeats.push_back(blockedRepeats[1]);
    blockedRepeats.erase(blockedRepeats.begin() + 1);

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    return optimizedDims.size() <= maxNDims;
}

std::vector<NodeDesc> TileBroadcastCommon::getSupportedConfigs(const Node *node, size_t outSize) {
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    auto precision = node->getOriginalInputPrecisionAtPort(0);
    auto dataType = DnnlExtensionUtils::ElementTypeToDataType(precision);

    const auto& srcDims = node->getInputShapeAtPort(0).getDims();
    const auto& inDataShape = node->getInputShapeAtPort(0);
    size_t outDataShapeRank = node->getOutputShapeAtPort(0).getRank();

    NodeConfig config;
    if (repeats.size() != outDataShapeRank && !repeats.empty())
        OPENVINO_THROW(node->getTypeStr(),
                       " node with name ",
                       node->getName(),
                       " has incorrect Repeats vector."
                       "Repeats rank must be equal to output shape rank. Repeats rank: ",
                       repeats.size(),
                       ", output shape rank: ",
                       outDataShapeRank);

    config.inConfs.resize(node->getParentEdges().size());
    config.inConfs[0].inPlace(-1);
    config.inConfs[0].constant(constMap[0]);
    config.inConfs[1].inPlace(-1);
    config.inConfs[1].constant(constMap[1]);
    config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, node->getInputShapeAtPort(1)));
    if (config.inConfs.size() == 3) {
        config.inConfs[2].inPlace(-1);
        config.inConfs[2].constant(constMap[2]);
        config.inConfs[2].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, node->getInputShapeAtPort(2)));
    }

    config.outConfs.resize(outSize);

    auto pushDesc = [&](dnnl::memory::format_tag inFormat, dnnl::memory::format_tag outFormat) {
        config.inConfs[0].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(node->getInputShapeAtPort(0), dataType, inFormat));
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].inPlace(-1);
            config.outConfs[i].constant(false);
            config.outConfs[i].setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(node->getOutputShapeAtPort(0), dataType, outFormat));
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    };

    if (!repeats.empty() && inDataShape.getRank() == outDataShapeRank && (outDataShapeRank == 4 || outDataShapeRank == 5)) {
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 16)) {
            if (outDataShapeRank == 4) {
                pushDesc(dnnl::memory::format_tag::nChw16c, dnnl::memory::format_tag::nChw16c);
            } else {
                pushDesc(dnnl::memory::format_tag::nCdhw16c, dnnl::memory::format_tag::nCdhw16c);
            }
        }
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 8)) {
            if (outDataShapeRank == 4) {
                pushDesc(dnnl::memory::format_tag::nChw8c, dnnl::memory::format_tag::nChw8c);
            } else {
                pushDesc(dnnl::memory::format_tag::nCdhw8c, dnnl::memory::format_tag::nCdhw8c);
            }
        }
        if (canBeExecutedInNSPCLayout(srcDims, repeats)) {
            if (outDataShapeRank == 4) {
                pushDesc(dnnl::memory::format_tag::nhwc, dnnl::memory::format_tag::nhwc);
            } else {
                pushDesc(dnnl::memory::format_tag::ndhwc, dnnl::memory::format_tag::ndhwc);
            }
        }
    }

    auto inFmt = DnnlExtensionUtils::GetPlainFormatByRank(inDataShape.getRank());
    auto outFmt = DnnlExtensionUtils::GetPlainFormatByRank(outDataShapeRank);
    if (inFmt == dnnl::memory::format_tag::undef || outFmt == dnnl::memory::format_tag::undef) {
        config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(precision, node->getInputShapeAtPort(0)));
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].inPlace(-1);
            config.outConfs[i].constant(false);
            config.outConfs[i].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(precision, node->getOutputShapeAtPort(i)));
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    } else {
        pushDesc(inFmt, outFmt);
    }

    return supportedPrimitiveDescriptors;
}

bool TileBroadcastCommon::prepareOptimizedParams(const Node *node, VectorDims& srcBlockedDims, VectorDims& dstBlockedDims) {
    while (srcBlockedDims.size() < dstBlockedDims.size()) {
        srcBlockedDims.insert(srcBlockedDims.begin(), 1);
    }

    VectorDims blockedRepeats = repeats;
    // for nC(d)hw16c and nC(d)hw8c layouts
    while (blockedRepeats.size() < dstBlockedDims.size()) {
        blockedRepeats.push_back(1);
    }
    // for NSPC layouts
    if (node->getBaseMemDescAtInputPort(0)->hasLayoutType(LayoutType::nspc) && one_of(node->getBaseMemDescAtInputPort(0)->getShape().getRank(), 4u, 5u)) {
        blockedRepeats.push_back(blockedRepeats[1]);
        blockedRepeats.erase(blockedRepeats.begin() + 1);
    }

    VectorDims optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    constexpr size_t maxNDims = 6lu;
    if (optimizedDims.size() > maxNDims)
        return false;

    while (optimizedDims.size() < maxNDims) {
        optimizedDims.insert(optimizedDims.begin(), 1);
        optimizedSrcStrides.insert(optimizedSrcStrides.begin(), 1);
    }

    VectorDims optimizedDstStrides = calculateDenseStrides(optimizedDims);

    size_t dataSize = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->getPrecision().size();
    for (size_t i = 0; i < optimizedDims.size(); i++) {
        optimizedSrcStrides[i] *= dataSize;
        optimizedDstStrides[i] *= dataSize;
    }

    optimizedParams.dims = optimizedDims;
    optimizedParams.srcStrides = optimizedSrcStrides;
    optimizedParams.dstStrides = optimizedDstStrides;
    optimizedParams.copySize = optimizedDims[5] * dataSize;

    return true;
}

// Broadcast 1 element to N continuous elements based on cpu_memcpy
// Step 1: Get the binary format of the number N
// Step 2: Use cpu_memcpy to form fragments containing pow(2, k) (ie. 2, 4, 8, ...) elements, based on the given 1 element
// Step 3: Form N continuous elements, who's a combination of those fragments, demonstrated by its binary format
void TileBroadcastCommon::broadcastScalar(const char *srcData, char *dstData, size_t elt_cnt, size_t data_size) {
    std::vector<size_t> binary_digits;

    binary_digits.clear();
    for (size_t tmp_cnt = elt_cnt; tmp_cnt > 0; tmp_cnt >>= 1) {
        binary_digits.emplace_back(tmp_cnt & 0x1);
    }

    size_t min_cnt = 1;
    size_t max_cnt = 1;
    auto curDstData = dstData;
    for (auto b : binary_digits) {
        if (b) {
            if (curDstData == dstData) {
                cpu_memcpy(curDstData, srcData, min_cnt * data_size);
            } else {
                cpu_memcpy(curDstData, dstData, min_cnt * data_size);
            }
            curDstData += min_cnt * data_size;
            for (size_t cur_cnt = min_cnt; cur_cnt < max_cnt; cur_cnt <<= 1) {
                cpu_memcpy(curDstData, dstData, cur_cnt * data_size);
                curDstData += cur_cnt * data_size;
            }
            min_cnt = max_cnt;
        }
        max_cnt <<= 1;
    }
}

void TileBroadcastCommon::optimizedExecute(const MemoryPtr& srcMemory, const MemoryPtr& dstMemory) {
    auto srcData = srcMemory->getDataAs<const char>();
    auto dstData = dstMemory->getDataAs<char>();

    if (srcMemory->getStaticDims() == dstMemory->getStaticDims()) {
        const auto prc = dstMemory->getDesc().getPrecision();
        // TODO: 109204
        // cpu_convert have to be used here because its implementation faster than cpu_memcpy
        // in the case when copySize exceeds L2 cache size
        cpu_convert(srcData, dstData, prc, prc, optimizedParams.copySize / prc.size());
    } else if (optimizedParams.srcStrides[5] == 0) {
        if (optimizedParams.dstStrides[0] == optimizedParams.dims[5] * optimizedParams.dstStrides[5]) {
            size_t data_size = optimizedParams.dstStrides[5];
            size_t elt_cnt = optimizedParams.dims[5];
            auto srcData_i32 = srcMemory->getDataAs<const int>();
            if (data_size == 1) {
                memset(dstData, srcData[0], elt_cnt);
            } else if (data_size == 4 && srcData_i32[0] == 0) {
                memset(dstData, 0, elt_cnt * data_size);
            } else {
                broadcastScalar(srcData, dstData, elt_cnt, data_size);
            }
        } else {
            parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                    [&](int i0, int i1, int i2, int i3, int i4) {
                auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                        i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                        i4 * optimizedParams.srcStrides[4]);
                auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                        i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                        i4 * optimizedParams.dstStrides[4]);
                for (size_t i = 0; i < optimizedParams.dims[5]; i++) {
                    cpu_memcpy(dstData2 + i * optimizedParams.dstStrides[5], srcData2, optimizedParams.dstStrides[5]);
                }
            });
        }
    } else {
        parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                [&](int i0, int i1, int i2, int i3, int i4) {
            auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                                                 i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                                                 i4 * optimizedParams.srcStrides[4]);
            auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                                           i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                                           i4 * optimizedParams.dstStrides[4]);
            cpu_memcpy(dstData2, srcData2, optimizedParams.copySize);
        });
    }
}

}   // namespace intel_cpu
}   // namespace ov
