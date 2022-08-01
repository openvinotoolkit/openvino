// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "grid_sample.hpp"
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "


bool GridSample::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v9::GridSample>(op)) {
            errorMessage = "Not supported GridSample operation version. CPU plug-in supports only 9th version.";
            return false;
        }
        if (!x64::mayiuse(x64::sse41)) {
            errorMessage = "Not supported CPU instructions set.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GridSample::GridSample(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges.";

    const auto& dataShape = getInputShapeAtPort(IN_DATA);
    if (dataShape.getRank() != 4)
        THROW_ERROR << "has incorrect rank of the Data input.";

    const auto& gridShape = getInputShapeAtPort(IN_GRID);
    if (gridShape.getRank() != 4)
        THROW_ERROR << "has incorrect rank of the Grid input.";
    if (gridShape.isStatic() && gridShape.getDims()[3] != 2)
        THROW_ERROR << "has incorrect shape of the Grid input. The 4th dimension should be equal to 2.";

    const auto& attributes = ov::as_type_ptr<ov::op::v9::GridSample>(op)->get_attributes();
    alignCorners = attributes.align_corners;
    switch (attributes.mode) {
        case op::v9::GridSample::InterpolationMode::BILINEAR:
            interpolationMode = InterpolationMode::BILINEAR;
            break;
        case op::v9::GridSample::InterpolationMode::BICUBIC:
            interpolationMode = InterpolationMode::BICUBIC;
            break;
        case op::v9::GridSample::InterpolationMode::NEAREST:
            interpolationMode = InterpolationMode::NEAREST;
            break;
    }
    switch (attributes.padding_mode) {
        case op::v9::GridSample::PaddingMode::ZEROS:
            paddingMode = PaddingMode::ZEROS;
            break;
        case op::v9::GridSample::PaddingMode::BORDER:
            paddingMode = PaddingMode::BORDER;
            break;
        case op::v9::GridSample::PaddingMode::REFLECTION:
            paddingMode = PaddingMode::REFLECTION;
            break;
    }
}

void GridSample::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataDims = getInputShapeAtPort(IN_DATA).getDims();

    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision.is_float()) {
        dataPrecision = Precision::FP32;
    } else {
        dataPrecision = Precision::I32;
    }
    dataTypeSize = dataPrecision.size();
    gridTypeSize = gridPrecision.size();

    impl_desc_type implType = jit_sse42;
    if (x64::mayiuse(x64::avx512_core)) {
        implType = jit_avx512;
    } else if (x64::mayiuse(x64::avx2)) {
        implType = jit_avx2;
    } else if (x64::mayiuse(x64::avx)) {
        implType = jit_avx;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, gridPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                         implType,
                         isDynamicNode());
}

void GridSample::createPrimitive() {
    jGridSampleConfParams jcp;

    jcp.inDataPrc     = dataPrecision;
    jcp.gridPrc       = gridPrecision;
    jcp.dynamicShapes = isDynamicNode();
    jcp.alignCorners  = alignCorners;
    jcp.interpolationMode = interpolationMode;
    jcp.paddingMode   = paddingMode;

    if (!jcp.dynamicShapes) {
        const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
        const auto& dstShape     = getOutputShapeAtPort(0).getDims();
        jcp.batchNum      = srcDataShape[0];
        jcp.srcBatchStepB = std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), dataTypeSize, std::multiplies<Dim>());
    }

    if (x64::mayiuse(x64::avx512_core)) {
        jitKernel.reset(new jitGridSampleKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        jitKernel.reset(new jitGridSampleKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::avx)) {
        jitKernel.reset(new jitGridSampleKernel<x64::avx>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        jitKernel.reset(new jitGridSampleKernel<x64::sse41>(jcp));
    }
    if (jitKernel) {
        jitKernel->create_ker();
    } else {
        THROW_ERROR << " could not create JIT kernel.";
    }

    execParamsPerThread.resize(parallel_get_max_threads());

    Node::createPrimitive();
}

void GridSample::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto& gridMemPtr = getParentEdgeAt(IN_GRID)->getMemoryPtr();
    if (!gridMemPtr || !gridMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input grid memory.";
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
    const auto& srcDataShape = dataMemPtr->getStaticDims();
    const auto& dstShape     = dstMemPtr->getStaticDims();
    const uint64_t totalWork = dstShape[2] * dstShape[3];
    const uint64_t nthr = parallel_get_max_threads();
    const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;

    parallel_nt(nthr, [&](const int ithr, const int nthr) {
        const uint64_t dstStart = std::min(wpt * ithr, totalWork);
        const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);

        auto& p = execParamsPerThread[ithr];

        p.batchNum    = srcDataShape[0];
        p.channelsNum = srcDataShape[1];
        p.srcHeightF  = srcDataShape[2];
        p.srcWidthF   = srcDataShape[3];

        p.workAmount = dstEnd - dstStart;
        p.gridStartB = dstStart * 2 * gridTypeSize;
        p.dstStartB  = dstStart * dataTypeSize;

        p.srcBatchStepB  = std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), dataTypeSize, std::multiplies<Dim>());
        p.gridBatchStepB = (dstShape[2] * dstShape[3] - p.workAmount) * 2 * gridTypeSize;
        p.dstBatchStepB  = (dstShape[1] * dstShape[2] * dstShape[3] - p.workAmount) * dataTypeSize;

        if (interpolationMode == InterpolationMode::BICUBIC && srcDataShape[3] >= 4) {
            p.srcWidthB = (srcDataShape[3] - 3) * dataTypeSize;
        } else {
            p.srcWidthB = srcDataShape[3] * dataTypeSize;
        }
        if (x64::mayiuse(x64::avx)) {
            p.srcHeightSub1F[0] = p.srcHeightF - 1.f;
            p.srcWidthSub1F[0]  = p.srcWidthF  - 1.f;
            p.srcHeightMul2F[0] = p.srcHeightF * 2.f;
            p.srcWidthMul2F[0]  = p.srcWidthF  * 2.f;
            if (alignCorners) {
                p.srcHeightMul2Sub1F[0] = p.srcHeightSub1F[0] * 2.f;
                p.srcWidthMul2Sub1F[0]  = p.srcWidthSub1F[0]  * 2.f;
                p.wDenormCoefF[0] = (p.srcWidthF  - 1.f) / 2.f;
                p.hDenormCoefF[0] = (p.srcHeightF - 1.f) / 2.f;
            } else {
                p.srcHeightMul2Sub1F[0] = p.srcHeightMul2F[0] - 1.f;
                p.srcWidthMul2Sub1F[0]  = p.srcWidthMul2F[0]  - 1.f;
            }
        } else {
            p.srcHeightSub1F = std::vector<float>(jitKernel->getDataElPerVec(), p.srcHeightF - 1.f);
            p.srcWidthSub1F  = std::vector<float>(jitKernel->getDataElPerVec(), p.srcWidthF  - 1.f);
            p.srcHeightMul2F = std::vector<float>(jitKernel->getDataElPerVec(), p.srcHeightF * 2.f);
            p.srcWidthMul2F  = std::vector<float>(jitKernel->getDataElPerVec(), p.srcWidthF  * 2.f);
            if (alignCorners) {
                p.srcHeightMul2Sub1F = std::vector<float>(jitKernel->getDataElPerVec(), p.srcHeightSub1F[0] * 2.f);
                p.srcWidthMul2Sub1F  = std::vector<float>(jitKernel->getDataElPerVec(), p.srcWidthSub1F[0]  * 2.f);
                p.wDenormCoefF = std::vector<float>(jitKernel->getDataElPerVec(), (p.srcWidthF  - 1.f) / 2.f);
                p.hDenormCoefF = std::vector<float>(jitKernel->getDataElPerVec(), (p.srcHeightF - 1.f) / 2.f);
            } else {
                p.srcHeightMul2Sub1F = std::vector<float>(jitKernel->getDataElPerVec(), p.srcHeightMul2F[0] - 1.f);
                p.srcWidthMul2Sub1F  = std::vector<float>(jitKernel->getDataElPerVec(), p.srcWidthMul2F[0]  - 1.f);
            }
        }
        p.srcChannelStepB = srcDataShape[2] * srcDataShape[3] * dataTypeSize;
        p.dstChannelStepB = dstShape[2] * dstShape[3] * dataTypeSize;
    });
}

void GridSample::execute(dnnl::stream strm) {
    const void* srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto threadBody = [&](const int ithr, const int nthr) {
        const auto& p = execParamsPerThread[ithr];
        auto arg = jGridSamplesExecArgs();

        arg.src                = srcData;
        arg.grid               = gridData + p.gridStartB;
        arg.dst                = dstData  + p.dstStartB;
        arg.batchNum           = p.batchNum;
        arg.channelsNum        = p.channelsNum;
        arg.srcHeightF         = &p.srcHeightF;
        arg.srcWidthF          = &p.srcWidthF;
        arg.srcWidthB          = &p.srcWidthB;
        arg.srcChannelStepB    = p.srcChannelStepB;
        arg.dstChannelStepB    = p.dstChannelStepB;
        arg.srcBatchStepB      = p.srcBatchStepB;
        arg.gridBatchStepB     = p.gridBatchStepB;
        arg.dstBatchStepB      = p.dstBatchStepB;
        arg.srcHeightSub1F     = p.srcHeightSub1F.data();
        arg.srcWidthSub1F      = p.srcWidthSub1F.data();
        arg.srcWidthMul2F      = p.srcWidthMul2F.data();
        arg.srcHeightMul2F     = p.srcHeightMul2F.data();
        arg.srcHeightMul2Sub1F = p.srcHeightMul2Sub1F.data();
        arg.srcWidthMul2Sub1F  = p.srcWidthMul2Sub1F.data();
        arg.wDenormCoefF       = p.wDenormCoefF.data();
        arg.hDenormCoefF       = p.hDenormCoefF.data();
        arg.workAmount         = p.workAmount;

        (*jitKernel)(&arg);
    };

    parallel_nt(0, threadBody);
}

void GridSample::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

std::vector<VectorDims> GridSample::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1, 2));
}

bool GridSample::created() const {
    return getType() == Type::GridSample;
}
