// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "gather.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include "kernels/gather_uni_kernel.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

namespace ov {
namespace intel_cpu {
namespace node {

bool Gather::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v7::Gather::get_type_info_static(),
                ov::op::v8::Gather::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 7 and 8 versions.";
            return false;
        }

        if (!isDynamicNgraphNode(op) && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
            errorMessage = "Only Constant operation on 'axis' input is supported for static node.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Gather::Gather(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache), batchDims(0) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_size() != 3 || op->get_output_size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges!";

    const auto& dataShape = getInputShapeAtPort(GATHER_DATA);
    isDataShapeStat = dataShape.isStatic();
    dataSrcRank = dataShape.getRank();

    const auto& idxShape = getInputShapeAtPort(GATHER_INDICES);
    isIdxShapeStat = idxShape.isStatic();
    const auto indicesRank = idxShape.getRank();
    if (dataSrcRank == 0lu || indicesRank == 0lu)
        THROW_ERROR << "has incorrect input parameters ranks.";

    if (ov::is_type<ov::op::v8::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v8::Gather>(op)->get_batch_dims());
        // WA for NMS->Gather construction. NMS fills part of the output blob by the -1 if these values
        // must not be taken into account. There is appropriate pass that looks for such subgraphs
        // and sets the dontReverseIndices flag.
        const auto& rti = op->get_rt_info();
        const auto& reverse = rti.find("dontReverseIndices");
        if (reverse == rti.end())
            reverseIndexing = true;
        else
            reverseIndexing = false;
    } else if (ov::is_type<ov::op::v7::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(op)->get_batch_dims());
        reverseIndexing = false;
    }

    if (batchDims < 0)
        batchDims += indicesRank;
    if (batchDims < 0 || batchDims > std::min(static_cast<int>(dataSrcRank), static_cast<int>(indicesRank)))
        THROW_ERROR << "has incorrect batch_dims " << batchDims << "!";

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR << "has incorrect input parameter axis value: " << axis;
    }
    shapeParameters.simdVecSize = x64::mayiuse(x64::avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen :
                           x64::mayiuse(x64::avx2) ? x64::cpu_isa_traits<x64::avx2>::vlen : 16;
}

void Gather::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    shapeParameters.dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
    shapeParameters.idxElPerVec = shapeParameters.simdVecSize / GatherShapeParameters::idxTypeSize;
    shapeParameters.dataElPerVec = shapeParameters.simdVecSize / shapeParameters.dataTypeSize;
    const auto& dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
    const auto& idxDims = getInputShapeAtPort(GATHER_INDICES).getDims();
    shapeParameters.initStatic(isAxisInputConst, isDataShapeStat, isIdxShapeStat, dataDims, idxDims, axis, batchDims);

    // Implementation desc type will be redefined in the fn prepareParams if a kernel will be created.
    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision}},
                         ref_any,
                         isDynamicNode());
}

void Gather::createOrUpdateJitKernelIfNeeded() {
    // Gather instruction is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_core) || x64::mayiuse(x64::avx2))) {
        x64::cpu_isa_t isa = x64::mayiuse(x64::avx512_core) ? x64::avx512_core : x64::avx2;
        jGatherConfParams jcp;
        jcp.dataTypeSize = shapeParameters.dataTypeSize;
        jcp.simdVecSize = shapeParameters.simdVecSize;
        jcp.idxElPerVec = shapeParameters.idxElPerVec;
        jcp.dataElPerVec = shapeParameters.dataElPerVec;
        jcp.reverseIndexing = reverseIndexing;
        jcp.dynamicShapes = isDynamicNode();
        jcp.batchDims = batchDims;
        jcp.beforeAxisSize = shapeParameters.beforeAxisSize;
        jcp.specIdxSize = shapeParameters.specIndicesSize;
        jcp.afterAxisSize = shapeParameters.afterAxisSize;

        if (jitKernel) {
            if (jitKernel->isSameParams(jcp)) {
                return;
            }
            jitKernel.reset();
        }
        jitKernel = jitGatherKernelInterface::createJitUniGatherKernel(
                isa, jcp.dataTypeSize, jcp.dynamicShapes, jcp.afterAxisSize, jcp.specIdxSize, shapeParameters.idxElPerVec);
        if (jitKernel) {
            jitKernel->initialize(jcp);
            jitKernel->create_ker();
        }
    }
}

void Gather::initializePerThreadParams() {
    if (!jitKernel) {
        return;
    }
    const uint64_t nthr = parallel_get_max_threads();
    const uint64_t workPerThread = ((shapeParameters.totalWork / shapeParameters.dataElPerVec) / nthr + 1) * shapeParameters.dataElPerVec;
    execParamsPerThread.resize(nthr);

    parallel_nt(nthr, [&](const int ithr, const int nthr) {
        const uint64_t dstStart = std::min(workPerThread * ithr, shapeParameters.totalWork);
        const uint64_t dstEnd = std::min(workPerThread * (ithr + 1), shapeParameters.totalWork);
        shapeParameters.fillPerThread(execParamsPerThread[ithr], dstStart, dstEnd);
    });
}

void Gather::createPrimitive() {
    Node::createPrimitive();
}

bool Gather::needPrepareParams() const {
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
    return result;
}

void Gather::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto& idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input indices memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR << "has incorrect input parameter axis value: " << axis;
    }

    const auto& dataDims = dataMemPtr->getStaticDims();
    const auto& idxDims = idxMemPtr->getStaticDims();
    shapeParameters.initDynamic(isAxisInputConst, isDataShapeStat, isIdxShapeStat, dataDims, idxDims, axis, batchDims);
    createOrUpdateJitKernelIfNeeded();
    initializePerThreadParams();
    const auto& selectedPD = getSelectedPrimitiveDescriptor();
    if (jitKernel) {
        if (x64::mayiuse(x64::avx512_core)) {
            selectedPD->setImplementationType(jit_avx512);
        } else if (x64::mayiuse(x64::avx2)) {
            selectedPD->setImplementationType(jit_avx2);
        }
    } else {
        selectedPD->setImplementationType(ref_any);
    }
}

void Gather::execute(dnnl::stream strm) {
    if (jitKernel) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

        auto threadBody = [&](const int ithr, const int nthr) {
            auto arg = shapeParameters.createArgStatic(dstData, srcData, srcIndices, execParamsPerThread[ithr]);
            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);
    } else {
        execReference();
    }
}

void Gather::executeDynamicImpl(dnnl::stream strm) {
    if (jitKernel) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

        auto threadBody = [&](const int ithr, const int nthr) {
            auto arg = shapeParameters.createArgDynamic(dstData, srcData, srcIndices, execParamsPerThread[ithr]);
            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);
    } else {
        execReference();
    }
}

void Gather::execReference() {
    const int32_t* srcIndices = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const size_t dstAfterBatchSize = shapeParameters.betweenBatchAndAxisSize * shapeParameters.specIdxAndAfterAxSizeB;
    parallel_for2d(shapeParameters.beforeBatchSize, shapeParameters.specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * shapeParameters.specIndicesSize + j];
        if (ii < 0) {
            if (reverseIndexing)
                ii += shapeParameters.axisDim;
            else
                ii = shapeParameters.axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + shapeParameters.afterAxisSizeInBytes * j;
        if (idx < shapeParameters.axisDim) {
            size_t c1 = shapeParameters.srcAfterBatchSizeInBytes * b + shapeParameters.afterAxisSizeInBytes * idx;
            for (size_t i = 0; i < shapeParameters.betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + shapeParameters.axisAndAfterAxisSizeInBytes * i;
                size_t dstIdx = c2 + shapeParameters.specIdxAndAfterAxSizeB * i;

                cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], shapeParameters.afterAxisSizeInBytes);
            }
        } else {
            for (size_t i = 0; i < shapeParameters.betweenBatchAndAxisSize; i++) {
                memset(&dstData[c2 + shapeParameters.specIdxAndAfterAxSizeB * i], 0, shapeParameters.afterAxisSizeInBytes);
            }
        }
    });
}

std::vector<VectorDims> Gather::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1, 2, 3));
}

bool Gather::created() const {
    return getType() == Type::Gather;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
