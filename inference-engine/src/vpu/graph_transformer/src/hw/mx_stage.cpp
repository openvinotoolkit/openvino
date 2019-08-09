// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/mx_stage.hpp>

#include <memory>
#include <vector>

#include <vpu/model/edges.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

StagePtr MyriadXHwStage::cloneImpl() const {
    return std::make_shared<MyriadXHwStage>(*this);
}

void MyriadXHwStage::propagateScaleFactorsImpl(const SmallVector<float>&, ScalePropagationStep) {
    VPU_THROW_EXCEPTION << "Must never be called";
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

void MyriadXHwStage::propagateDataOrderImpl() const {
    IE_ASSERT(_inputEdges.size() >= 4);
    IE_ASSERT(_outputEdges.size() >= 1);

    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();
        auto scales = _inputEdges[3]->input();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);
        IE_ASSERT(scales->usage() == DataUsage::Const || scales->usage() == DataUsage::Fake);
    }

    auto input = _inputEdges[0]->input();
    auto output = _outputEdges[0]->output();

    // TODO: support HCW

    if (input->desc().numDims() >= 3) {
        _orderInfo.setInput(_inputEdges[0], input->desc().dimsOrder().createMovedDim(Dim::C, 2));
    } else {
        IE_ASSERT(input->desc().dimsOrder() == DimsOrder::NC);
    }

    if (output->desc().numDims() >= 3) {
        _orderInfo.setOutput(_outputEdges[0], output->desc().dimsOrder().createMovedDim(Dim::C, 2));
    } else {
        IE_ASSERT(output->desc().dimsOrder() == DimsOrder::NC);
    }
}

void MyriadXHwStage::getDataStridesRequirementsImpl() const {
    IE_ASSERT(_inputEdges.size() >= 4);
    IE_ASSERT(_outputEdges.size() >= 1);

    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();
        auto scales = _inputEdges[3]->input();

        IE_ASSERT(weights->usage() == DataUsage::Const);
        IE_ASSERT(biases->usage() == DataUsage::Const || biases->usage() == DataUsage::Fake);
        IE_ASSERT(scales->usage() == DataUsage::Const || scales->usage() == DataUsage::Fake);
    }

    auto input = _inputEdges[0]->input();
    auto output = _outputEdges[0]->output();

    _stridesInfo.setInput(_inputEdges[0], getHwStridesRequirement(handle_from_this(), input->desc()));
    _stridesInfo.setOutput(_outputEdges[0], getHwStridesRequirement(handle_from_this(), output->desc()));
}

void MyriadXHwStage::finalizeDataLayoutImpl() {
}

void MyriadXHwStage::getBatchSupportInfoImpl() const {
    if (attrs().get<HwOpType>("hwOpType") != HwOpType::POOL) {
        IE_ASSERT(_inputEdges.size() >= 4);
        IE_ASSERT(_outputEdges.size() >= 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }
}

void MyriadXHwStage::finalCheckImpl() const {
    IE_ASSERT(_inputEdges.size() >= 4);
    IE_ASSERT(_outputEdges.size() >= 1);

    auto input = _inputEdges[0]->input();
    auto weights = _inputEdges[1]->input();
    auto biases = _inputEdges[2]->input();
    auto scales = _inputEdges[3]->input();
    auto output = _outputEdges[0]->output();

    IE_ASSERT(input->memoryOffset() % 16 == 0);
    IE_ASSERT(weights->memoryOffset() % 16 == 0);
    IE_ASSERT(biases->memoryOffset() % 16 == 0);
    IE_ASSERT(scales->memoryOffset() % 16 == 0);
    IE_ASSERT(output->memoryOffset() % 16 == 0);
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

    serializer.append(checked_cast<uint32_t>(_injectedStageEdges.size()));
    for (const auto& injectedStageEdge : _injectedStageEdges) {
        injectedStageEdge->child()->serialize(serializer);
    }
}

void MyriadXHwStage::serializeDataImpl(BlobSerializer& serializer) const {
    auto numBuffersPos = serializer.append(static_cast<uint32_t>(0));

    uint32_t numBuffers = 0;

    for (const auto& inEdge : _inputEdges) {
        if (inEdge->childEdge() != nullptr)
            continue;

        if (inEdge->input()->usage() == DataUsage::Fake)
            continue;

        inEdge->input()->serializeNewBuffer(serializer);

        ++numBuffers;
    }

    for (const auto& outEdge : _outputEdges) {
        if (outEdge->childEdge() != nullptr)
            continue;

        if (outEdge->output()->usage() == DataUsage::Fake)
            continue;

        outEdge->output()->serializeNewBuffer(serializer);

        ++numBuffers;
    }

    serializer.overWrite(numBuffersPos, checked_cast<uint32_t>(numBuffers));
}

}  // namespace vpu
