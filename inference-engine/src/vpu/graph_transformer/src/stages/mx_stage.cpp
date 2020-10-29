// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/mx_stage.hpp>

#include <memory>
#include <vector>

#include <vpu/model/edges.hpp>
#include <vpu/middleend/hw/utility.hpp>

namespace vpu {

StagePtr MyriadXHwStage::cloneImpl() const {
    return std::make_shared<MyriadXHwStage>(*this);
}

namespace {

StridesRequirement getHwStridesRequirement(const Stage& stage, const DataDesc& desc) {
    StridesRequirement out;

    if (desc.numDims() >= 3) {
        out.add(1, DimStride::Aligned);
    } else {
        IE_ASSERT(stage->attrs().get<HwOpType>("hwOpType") == HwOpType::FC);
        IE_ASSERT(desc.dimsOrder() == DimsOrder::NC);

        out.add(0, DimStride::Aligned);
    }

    if (desc.dim(Dim::N, 1) > 1) {
        // To merge batch into previous dimension.
        out.add(desc.dimsOrder().dimInd(Dim::N), DimStride::Compact);
    }

    return out;
}

}  // namespace

void MyriadXHwStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        auto weights = inputEdge(1)->input();
        auto biases = inputEdge(2)->input();
        auto scales = inputEdge(3)->input();

        IE_ASSERT(weights->usage() == DataUsage::Const || weights->usage() == DataUsage::Intermediate);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);
        IE_ASSERT(scales->usage() == DataUsage::Const || scales->usage() == DataUsage::Fake);
    }

    auto input = inputEdge(0)->input();
    auto output = outputEdge(0)->output();

    // TODO: support HCW

    if (input->desc().numDims() >= 3) {
        orderInfo.setInput(inputEdge(0), input->desc().dimsOrder().createMovedDim(Dim::C, 2));
    } else {
        IE_ASSERT(input->desc().dimsOrder() == DimsOrder::NC);
    }

    if (output->desc().numDims() >= 3) {
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 2));
    } else {
        IE_ASSERT(output->desc().dimsOrder() == DimsOrder::NC);
    }
}

void MyriadXHwStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        auto weights = inputEdge(1)->input();
        auto biases = inputEdge(2)->input();
        auto scales = inputEdge(3)->input();

        IE_ASSERT(weights->usage() == DataUsage::Const || weights->usage() == DataUsage::Intermediate);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);
        IE_ASSERT(scales->usage() == DataUsage::Const || scales->usage() == DataUsage::Fake);
    }

    auto input = inputEdge(0)->input();
    auto output = outputEdge(0)->output();

    stridesInfo.setInput(inputEdge(0), getHwStridesRequirement(this, input->desc()));
    stridesInfo.setOutput(outputEdge(0), getHwStridesRequirement(this, output->desc()));
}

void MyriadXHwStage::finalizeDataLayoutImpl() {
}

void MyriadXHwStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }
}

void MyriadXHwStage::finalCheckImpl() const {
    const auto input = inputEdge(0)->input();
    const auto output = outputEdge(0)->output();

    IE_ASSERT(input->memoryOffset() % 16 == 0);
    IE_ASSERT(output->memoryOffset() % 16 == 0);

    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        const auto weights = inputEdge(1)->input();
        const auto biases = inputEdge(2)->input();
        const auto scales = inputEdge(3)->input();

        IE_ASSERT(weights->memoryOffset() % 16 == 0);
        IE_ASSERT(biases->memoryOffset() % 16 == 0);
        IE_ASSERT(scales->memoryOffset() % 16 == 0);
    }
}

void MyriadXHwStage::serializeParamsImpl(BlobSerializer& serializer) const {
    const auto& hwOps = attrs().get<HwOpList>("hwOps");
    IE_ASSERT(!hwOps.vec.empty());

    serializer.append(checked_cast<uint32_t>(hwOps.vec.size()));
    for (const auto& hwOpParams : hwOps.vec) {
        serializer.append(checked_cast<uint32_t>(hwOpParams.opType));
        if (hwOpParams.opType == HwOpType::POOL) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.poolType));
        }

        serializer.append(checked_cast<uint32_t>(hwOpParams.opMode));

        serializer.append(checked_cast<uint32_t>(hwOpParams.withPad));
        if (hwOpParams.withPad) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.padMode));
        }

        serializer.append(checked_cast<int32_t>(hwOpParams.inputInd));
        serializer.append(checked_cast<int32_t>(hwOpParams.outputInd));
        serializer.append(checked_cast<int32_t>(hwOpParams.coeffsInd));
        serializer.append(checked_cast<int32_t>(hwOpParams.biasesInd));
        serializer.append(checked_cast<int32_t>(hwOpParams.scalesInd));

        if (hwOpParams.opType != HwOpType::FC) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.outChanOffset));
            serializer.append(checked_cast<uint32_t>(hwOpParams.outNumChans));
        } else {
            serializer.append(checked_cast<uint32_t>(hwOpParams.fcInputOffset));
            serializer.append(checked_cast<uint32_t>(hwOpParams.fcInputNum));
            serializer.append(checked_cast<uint32_t>(hwOpParams.fcOutputOffset));
            serializer.append(checked_cast<uint32_t>(hwOpParams.fcOutputNum));
            serializer.append(checked_cast<uint32_t>(hwOpParams.fcAccum));
        }

        if (hwOpParams.opType != HwOpType::FC) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.kernelWidth));
            serializer.append(checked_cast<uint32_t>(hwOpParams.kernelHeight));
            serializer.append(checked_cast<uint32_t>(hwOpParams.kernelStride));
        }

        if (hwOpParams.opType == HwOpType::CONV_POOL) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.poolKernelWidth));
            serializer.append(checked_cast<uint32_t>(hwOpParams.poolKernelHeight));
        }

        serializer.append(checked_cast<uint32_t>(hwOpParams.withReLU));
        if (hwOpParams.withReLU) {
            serializer.append(checked_cast<uint32_t>(hwOpParams.t0));
            serializer.append(checked_cast<uint32_t>(hwOpParams.a0));
            serializer.append(checked_cast<uint32_t>(hwOpParams.a1));
        }

        serializer.append(checked_cast<uint32_t>(hwOpParams.withClamp));
        if (hwOpParams.withClamp) {
            serializer.append(checked_cast<float>(hwOpParams.clampMaxVal));
        }

        serializer.append(checked_cast<uint32_t>(hwOpParams.reuseData));
        serializer.append(checked_cast<uint32_t>(hwOpParams.reuseCoeff));
    }

    serializer.append(checked_cast<uint32_t>(injectedStage() == nullptr ? 0 : 1));
    if (const auto injectedStage = this->injectedStage()) {
        injectedStage->serialize(serializer);
    }
}

void MyriadXHwStage::serializeDataImpl(BlobSerializer& serializer) const {
    auto numBuffersPos = serializer.append(static_cast<uint32_t>(0));

    uint32_t numBuffers = 0;

    for (const auto& inEdge : inputEdges()) {
        if (inEdge->childEdge() != nullptr)
            continue;

        if (inEdge->input()->usage() == DataUsage::Fake)
            continue;

        inEdge->input()->serializeBuffer(serializer);

        ++numBuffers;
    }

    for (const auto& outEdge : outputEdges()) {
        if (outEdge->childEdge() != nullptr)
            continue;

        if (outEdge->output()->usage() == DataUsage::Fake)
            continue;

        outEdge->output()->serializeBuffer(serializer);

        ++numBuffers;
    }

    serializer.overWrite(numBuffersPos, checked_cast<uint32_t>(numBuffers));
}

}  // namespace vpu
