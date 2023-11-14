// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "gather.h"
#include <openvino/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include "kernels/x64/gather_uni_kernel.hpp"
#include <partitioned_mem_mgr.h>
#include "shape_inference/custom/gather.hpp"

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

Gather::Gather(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, GatherShapeInferFactory(op)),
      batchDims(0) {
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

    if (auto indices = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_INDICES))) {
        constIndices = indices->cast_vector<int>();
    }
}

void Gather::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();

    const auto& dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
    if (isAxisInputConst && isDataShapeStat) {
        axisDim = dataDims[axis];
        beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
        betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    }
    if (isDataShapeStat) {
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<Dim>());
    }
    if (isIdxShapeStat) {
        const auto& idxDims = getInputShapeAtPort(GATHER_INDICES).getDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<Dim>());

        if (isDataShapeStat) {
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    // Implementation desc type will be redefined in the fn prepareParams if a kernel will be created.
    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision}},
                         ref_any);

    // Let's check for the special inPlace memory use case
    // in place only makes sense when we split by dense blocks since strided tensors are not supported by most nodes

    if (!isAxisInputConst) {
        return;
    }

    if (batchDims != 0) {
        return;
    }

    if (constIndices.size() != 1) {
        return;
    }

    const auto& parentDims = inputShapes[0].getDims();
    const auto axisDim = parentDims[axis];
    if (Shape::UNDEFINED_DIM == axisDim) {
        return;
    }

    const auto indx = constIndices.front();
    const auto normIndex = indx < 0 ? static_cast<int64_t>(axisDim) + indx : indx;

    if (normIndex < 0 || normIndex >= static_cast<int64_t>(axisDim)) {
        return;
    }

    if (std::any_of(parentDims.begin(), parentDims.begin() + axis, [](size_t dim) { return  dim != 1; })) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                    {LayoutType::ncsp, ov::element::i32},
                    {LayoutType::ncsp, ov::element::i32, isAxisInputConst}},
                    {{LayoutType::ncsp, dataPrecision, false, GATHER_DATA}},
                    unknown);
}

void Gather::createPrimitive() {
    if (isInPlace()) {
        return;
    }
#if defined(OPENVINO_ARCH_X86_64)
    uint64_t idxElPerVec = 1;
    if (!isDynamicNode()) {
        idxElPerVec = x64::mayiuse(x64::avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen / idxTypeSize :
            x64::mayiuse(x64::avx2) ? x64::cpu_isa_traits<x64::avx2>::vlen / idxTypeSize : 1;
    }
    // Gather instruction is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_core) || x64::mayiuse(x64::avx2)) &&
            (isDynamicNode() || afterAxisSize == 1 || (afterAxisSize <= idxElPerVec &&
            (x64::mayiuse(x64::avx512_core) || (x64::mayiuse(x64::avx2) && dataTypeSize == 4))))) {
        jGatherConfParams jcp;
        jcp.dataTypeSize = dataTypeSize;
        jcp.reverseIndexing = reverseIndexing;
        jcp.dynamicShapes = isDynamicNode();
        jcp.batchDims = batchDims;
        if (!jcp.dynamicShapes) {
            jcp.beforeAxisSize = beforeAxisSize;
            jcp.specIdxSize = specIndicesSize;
            jcp.afterAxisSize = afterAxisSize;
        } else {
            if (isDataShapeStat && isAxisInputConst) {
                jcp.beforeAxisSize = beforeAxisSize;
                jcp.afterAxisSize = afterAxisSize;
            }
            if (isIdxShapeStat) {
                jcp.specIdxSize = specIndicesSize;
            }
        }

        if (x64::mayiuse(x64::avx512_core)) {
            jitKernel.reset(new jitUniGatherKernel<x64::avx512_core>(jcp));
        } else if (x64::mayiuse(x64::avx2)) {
            jitKernel.reset(new jitUniGatherKernel<x64::avx2>(jcp));
        }
        if (jitKernel) {
            jitKernel->create_ker();

            if (!isDynamicNode()) {
                const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
                const uint64_t nthr = parallel_get_max_threads();
                const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
                execParamsPerThread.resize(nthr);

                parallel_nt(nthr, [&](const int ithr, const int nthr) {
                    const uint64_t dstStart = std::min(wpt * ithr, totalWork);
                    const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);

                    auto& p = execParamsPerThread[ithr];
                    p.workAmount = dstEnd - dstStart;
                    p.dstStart = dstStart;
                    p.specIdxInBytes.resize(dataElPerVec);
                    p.idxBatchSumInBytes.resize(dataElPerVec);
                    p.dataBeforeAxisSumInBytes.resize(dataElPerVec);
                    p.betweenBatchAndAxisIter = (dstStart / specIndicesSize) % betweenBatchAndAxisSize;
                    for (uint64_t j = 0lu; j < dataElPerVec; j++) {
                        p.specIdxInBytes[j] = (((dstStart + j) / afterAxisSize) % specIndicesSize) * idxTypeSize;
                        p.idxBatchSumInBytes[j] = ((dstStart + j) / (betweenBatchAndAxisSize * specIndicesSize * afterAxisSize)) *
                                specIndicesSize * idxTypeSize;
                        p.dataBeforeAxisSumInBytes[j] = ((dstStart + j) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;
                    }
                    initShortParams(p, dstStart);
                });
            }
        }
    }
#endif
    Node::createPrimitive();
}

bool Gather::needPrepareParams() const {
    if (isInPlace()) {
        return false;
    }
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->getData()))[0];
    return result;
}

void Gather::prepareParams() {
    auto dataMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input indices memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->getData()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR << "has incorrect input parameter axis value: " << axis;
    }

    if (!isDataShapeStat || !isAxisInputConst) {
        const auto& dataDims = dataMemPtr->getStaticDims();
        axisDim = dataDims[axis];
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
        betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;

        if (isIdxShapeStat) {
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    if (!isIdxShapeStat) {
        const auto& idxDims = idxMemPtr->getStaticDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

        specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
        totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
    }

#if defined(OPENVINO_ARCH_X86_64)
    const auto& selectedPD = getSelectedPrimitiveDescriptor();
    if (jitKernel && jitKernel->isSupportedConfiguration(afterAxisSize)) {
        if (x64::mayiuse(x64::avx512_core)) {
            selectedPD->setImplementationType(jit_avx512);
        } else if (x64::mayiuse(x64::avx2)) {
            selectedPD->setImplementationType(jit_avx2);
        }
    }
#endif
}

void Gather::execute(dnnl::stream strm) {
    if (isInPlace()) {
        return;
    }
#if defined(OPENVINO_ARCH_X86_64)
    if (jitKernel && jitKernel->isSupportedConfiguration(afterAxisSize)) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->getData();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getData();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->getData());

        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            auto& p = execParamsPerThread[ithr];
            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + p.dstStart * dataTypeSize;
            arg.indices = srcIndices;
            arg.start = &p.dstStart;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = p.workAmount;
            arg.specIdxB = p.specIdxInBytes.data();
            arg.idxBatchSumB = p.idxBatchSumInBytes.data();
            arg.dataBeforeAxisSumB = p.dataBeforeAxisSumInBytes.data();
            arg.betweenBatchAndAxisIter = p.betweenBatchAndAxisIter;

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();

            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) { // Elementwise short case.
                arg.permIdxMask = p.permIdxMask.data();
                arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
            } else if (afterAxisSize > 1 && afterAxisSize <= dataElPerVec) { // Blocked short case.
                arg.afterAxIdxB = p.afterAxIdxInBytes.data();
                arg.specIdxDiff = p.specIdxDiff.data();
                arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
                arg.beforeAxisPermMask = p.beforeAxPermMask.data();
                arg.afterAxisPermMask = p.afterAxPermMask.data();
                arg.afterAxisSize = &afterAxisSize;
                arg.specIdxAndAfterAxIterB = p.specIdxAndAfterAxIterB;
                arg.specIdxAndAfterAxSizeB = specIdxAndAfterAxSizeB;
            }

            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);

        return;
    }
#endif
    execReference();
}

void Gather::executeDynamicImpl(dnnl::stream strm) {
    if (isInPlace()) {
        return;
    }
#if defined(OPENVINO_ARCH_X86_64)
    if (jitKernel && jitKernel->isSupportedConfiguration(afterAxisSize)) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->getData();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getData();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->getData());

        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
            const uint64_t workAmount = end - start;

            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + afterAxisSizeInBytes * start;
            arg.indices = srcIndices;
            arg.start = &start;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = workAmount;

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
            int permIdxMask[16];
            int beforeAxisDiff[16];
            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
                permIdxMask[0] = idxElPerVec - specIndicesSize;
                int div = idxElPerVec / specIndicesSize;
                int remainder = idxElPerVec % specIndicesSize;
                for (uint64_t i = 1; i < idxElPerVec; i++) {
                    permIdxMask[i] = permIdxMask[i - 1] + 1;
                    if (static_cast<uint64_t>(permIdxMask[i]) == idxElPerVec)
                        permIdxMask[i] = idxElPerVec - specIndicesSize;
                }
                for (uint64_t i = 0; i < idxElPerVec; i++) {
                    if (((start + i) % specIndicesSize) < (specIndicesSize - remainder))
                        beforeAxisDiff[i] = axisDim * div;
                    else
                        beforeAxisDiff[i] = axisDim * (div + 1);
                }
                arg.permIdxMask = permIdxMask;
                arg.beforeAxisDiff = beforeAxisDiff;
            }

            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);

        return;
    }
#endif
    execReference();
}

void Gather::initShortParams(threadExecParams& p, const uint64_t start) {
    if (!jitKernel)
        THROW_ERROR << "has uninitialized kernel in function initShortParams.";
    const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();

    if (afterAxisSize == 1) { // Elementwise gather.
        if (specIndicesSize >= idxElPerVec)
            return; // Is not a short case.

        p.permIdxMask.resize(idxElPerVec);
        p.srcBeforeAxisDiff.resize(idxElPerVec);

        p.permIdxMask[0] = idxElPerVec - specIndicesSize;
        for (uint64_t i = 1; i < idxElPerVec; i++) {
            p.permIdxMask[i] = p.permIdxMask[i - 1] + 1;
            if (static_cast<uint64_t>(p.permIdxMask[i]) == idxElPerVec)
                p.permIdxMask[i] = idxElPerVec - specIndicesSize;
        }

        const int div = idxElPerVec / specIndicesSize;
        const int remainder = idxElPerVec % specIndicesSize;
        for (uint64_t i = 0; i < idxElPerVec; i++) {
            if (((start + i) % specIndicesSize) < (specIndicesSize - remainder)) {
                p.srcBeforeAxisDiff[i] = axisDim * div;
            } else {
                p.srcBeforeAxisDiff[i] = axisDim * (div + 1);
            }
        }
    } else { // Blocked gather.
        if (afterAxisSize > idxElPerVec)
            return; // Is not a short case.

        p.afterAxIdxInBytes.resize(idxElPerVec);
        p.afterAxPermMask.resize(idxElPerVec);
        p.beforeAxPermMask.resize(idxElPerVec);
        p.specIdxDiff.resize(idxElPerVec);
        p.srcBeforeAxisDiff.resize(idxElPerVec);

        int secondStart = start + idxElPerVec;
        for (uint64_t i = 0; i < idxElPerVec; i++) {
            p.afterAxIdxInBytes[i] = (start + i) % afterAxisSize;
            p.specIdxDiff[i] = (((secondStart + i) / afterAxisSize) % specIndicesSize) * idxTypeSize - p.specIdxInBytes[i];
            if (p.specIdxDiff[i] < 0)
                p.specIdxDiff[i] += specIndicesSize * idxTypeSize;
            p.srcBeforeAxisDiff[i] = ((start + i + idxElPerVec) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes -
                    ((start + i) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;

            p.afterAxIdxInBytes[i] *= dataTypeSize;
            p.afterAxPermMask[i] = idxElPerVec - afterAxisSize + i;
            for (size_t j = 0lu; j < 6lu; j++) {
                if (static_cast<uint64_t>(p.afterAxPermMask[i]) >= idxElPerVec)
                    p.afterAxPermMask[i] -= afterAxisSize;
            }
        }
        if (specIndicesSize * afterAxisSize < idxElPerVec) {
            p.beforeAxPermMask[0] = idxElPerVec - specIndicesSize * afterAxisSize;
            for (uint64_t i = 1; i < idxElPerVec; i++) {
                p.beforeAxPermMask[i] = p.beforeAxPermMask[i - 1] + 1;
                if (static_cast<uint64_t>(p.beforeAxPermMask[i]) == idxElPerVec)
                    p.beforeAxPermMask[i] = idxElPerVec - specIndicesSize * afterAxisSize;
            }
        }

        p.specIdxAndAfterAxIterB = (start * dataTypeSize) % specIdxAndAfterAxSizeB;
    }
}

void Gather::execReference() {
    const int32_t* srcIndices = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->getData());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getData());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->getData());

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;
    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
        if (idx < static_cast<size_t>(axisDim)) {
            size_t c1 = srcAfterBatchSizeInBytes * b + afterAxisSizeInBytes * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSizeInBytes * i;
                size_t dstIdx = c2 + specIdxAndAfterAxSizeB * i;

                cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                memset(&dstData[c2 + specIdxAndAfterAxSizeB * i], 0, afterAxisSizeInBytes);
            }
        }
    });
}

bool Gather::created() const {
    return getType() == Type::Gather;
}

bool Gather::isExecutable() const {
    return !isInPlace() && Node::isExecutable();
}

void Gather::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP) || !isInPlace()) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    constexpr size_t outputPort = 0;

    auto& config = selected_pd->getConfig();
    size_t inplaceInpIndx = selected_pd->getConfig().outConfs[outputPort].inPlace();
    const auto baseDim = inputShapes.front().getDims()[axis];
    IE_ASSERT(baseDim != Shape::UNDEFINED_DIM) << "Gather node: " << getName() << " can not use inPlace memory with splitting on dynamic dimention";
    auto baseMemMngr = getParentEdgesAtPort(inplaceInpIndx).front()->getMemory().getMemoryMngr();
    const auto index = constIndices.front();
    const ptrdiff_t offset = index < 0 ? baseDim + index : index;
    const auto& childEdges = getChildEdgesAtPort(outputPort);
    for (auto& childEdge : childEdges) {
        IE_ASSERT(childEdge->getStatus() == Edge::Status::NotAllocated) << " Unexpected edge status in node: " <<
            getName() << " with type " << getTypeStr();

        auto memMngr = std::make_shared<PartitionedMemoryMngr>(baseMemMngr, baseDim, offset);
        auto newMem = std::make_shared<Memory>(getEngine(), config.outConfs[outputPort].getMemDesc(), memMngr);

        childEdge->reuse(newMem);
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
