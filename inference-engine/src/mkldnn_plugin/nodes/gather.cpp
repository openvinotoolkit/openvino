// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include "ie_parallel.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "common/cpu_memcpy.h"
#include <mkldnn_types.h>
#include <string>
#include <vector>

#include <chrono> // remove

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using namespace mkldnn::impl::cpu;

struct jGatherConfParams {
    int32_t beforeAxisSize;
    int32_t indicesSize;
    uint32_t dictTypeSize;
};

struct jGatherArgs {
    const void* src;
    void* dst;
    const int* indices;
    const int* dictTypeSize;
    const int* axisDim;
    const int* axDimSum;
    const int* shufMask8bit;
    const int* permMask8bit;
    const int* shufMask16bit;
    const int* permMask16bit;
    size_t idxStartB;
    size_t workAmount;
    int* tmp; // remove
    int* retVal; // remove
};

struct jitGatherKernelBase {
    void (*ker_)(const jGatherArgs *);
    void operator()(const jGatherArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGatherKernelBase(jGatherConfParams jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jitGatherKernelBase() {}

    virtual void create_ker() = 0;

    jGatherConfParams jcp_;
};

#define GET_OFF(field) offsetof(jGatherArgs, field)

template <x64::cpu_isa_t isa>
struct jitUniGatherKernel : public jitGatherKernelBase, public x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel)

    explicit jitUniGatherKernel(jGatherConfParams jcp) : jitGatherKernelBase(jcp), x64::jit_generator() {}

    void create_ker() override {
        x64::jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(regSrc, ptr[regParams + GET_OFF(src)]);
        mov(regDst, ptr[regParams + GET_OFF(dst)]);
        mov(regIndices, ptr[regParams + GET_OFF(indices)]);

        mov(regIdxIter, ptr[regParams + GET_OFF(idxStartB)]);

        mov(regAux1, ptr[regParams + GET_OFF(dictTypeSize)]);
        uni_vpbroadcastd(vmmDictTypeSize, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxDim, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axDimSum)]);
        uni_vpbroadcastd(vmmAxDimSum, ptr[regAux1]);

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

//        mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//        mov(regAux2, ptr[regParams + GET_OFF(retVal)]);

        elPerVec = vlen / jcp_.dictTypeSize;

        if (jcp_.dictTypeSize == 4) {
            Xbyak::Label lDstIdxLoop, lTail, lFinish;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                fillIndicies(vmmSrcShifts);
                uni_vpcmpeqd(vmmOnesBit, vmmOnesBit, vmmOnesBit);
                vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmOnesBit);
                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            Xbyak::Label lTailLoop, lCalc;
            Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
            Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
            Xbyak::Reg32 regAux3_32(regAux3.getIdx());
            uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
            uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
            L(lTailLoop);
            {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                cmp(regIdxIter, jcp_.indicesSize);
                jl(lCalc, T_NEAR);
                mov(regIdxIter, 0);
                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
                uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);

                L(lCalc);
                mov(eax, ptr[regIndices + regIdxIter]);
                mul(regDictTypeSize32);
                add(eax, regAxDimSum32);
                mov(regAux3_32, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_32);

                add(regIdxIter, sizeof(int));
                add(regDst, jcp_.dictTypeSize);
                sub(regWorkAmount, 1);
                jmp(lTailLoop, T_NEAR);
            }

            L(lFinish);
        } else if (jcp_.dictTypeSize == 2) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask16bit)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            mov(regAux1, ptr[regParams + GET_OFF(permMask16bit)]);
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail, lFinish;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                gatherAndShift(vmmAux3, vmmShufMask); // VPERMB 512
                gatherAndShift(vmmAux4, vmmShufMask);
                vshufps(vmmAux3, vmmAux3, vmmAux4, 0x44);
                vpermd(vmmAux3, vmmPermMask, vmmAux3);

                uni_vmovups(ptr[regDst], vmmAux3);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            Xbyak::Label lTailLoop, lCalc;
            Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
            Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
            Xbyak::Reg16 regAux3_16(regAux3.getIdx());
            uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
            uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
            L(lTailLoop);
            {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                cmp(regIdxIter, jcp_.indicesSize);
                jl(lCalc, T_NEAR);
                mov(regIdxIter, 0);
                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
                uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);

                L(lCalc);
                mov(eax, ptr[regIndices + regIdxIter]);
                mul(regDictTypeSize32);
                add(eax, regAxDimSum32);
                mov(regAux3_16, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_16);

                add(regIdxIter, sizeof(int));
                add(regDst, jcp_.dictTypeSize);
                sub(regWorkAmount, 1);
                jmp(lTailLoop, T_NEAR);
            }
            L(lFinish);
        } else if (jcp_.dictTypeSize == 1) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask8bit)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            mov(regAux1, ptr[regParams + GET_OFF(permMask8bit)]);
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail, lFinish;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                gatherAndShift(vmmAux3, vmmShufMask);
                gatherAndShift(vmmAux4, vmmShufMask);
                vshufps(vmmAux3, vmmAux3, vmmAux4, 0);

                gatherAndShift(vmmAux4, vmmShufMask);
                gatherAndShift(vmmAux5, vmmShufMask);
                vshufps(vmmAux4, vmmAux4, vmmAux5, 0);

                vshufps(vmmAux3, vmmAux3, vmmAux4, 0x88);
                vpermd(vmmAux3, vmmPermMask, vmmAux3);

                uni_vmovups(ptr[regDst], vmmAux3);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            Xbyak::Label lTailLoop, lCalc;
            Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
            Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
            Xbyak::Reg8 regAux3_8(regAux3.getIdx());
            uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
            uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
            L(lTailLoop);
            {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                cmp(regIdxIter, jcp_.indicesSize);
                jl(lCalc, T_NEAR);
                mov(regIdxIter, 0);
                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
                uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);

                L(lCalc);
                mov(eax, ptr[regIndices + regIdxIter]);
                add(eax, regAxDimSum32);
                mov(regAux3_8, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_8);

                add(regIdxIter, sizeof(int));
                add(regDst, jcp_.dictTypeSize);
                sub(regWorkAmount, 1);
                jmp(lTailLoop, T_NEAR);
            }
            L(lFinish);
        }

        this->postamble();
    }

protected:
    void fillIndicies(Xbyak::Xmm& dst) {
        Xbyak::Label lPerElements, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlenXmm);
        jg(lPerElements, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, xmmDictTypeSize);
            uni_vpaddd(dst, dst, xmmAxDimSum);
            add(regIdxIter, vlenXmm);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);

        L(lPerElements);
        for (uint8_t i = 0; i < 4; i++) {
            Xbyak::Label insertLabel;

            cmp(regIdxIter, jcp_.indicesSize);
            jl(insertLabel, T_NEAR);
                mov(regIdxIter, 0);
                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);

            L(insertLabel);
            uni_vpbroadcastd(xmmAux1, ptr[regIndices + regIdxIter]);
            uni_vpmulld(xmmAux1, xmmAux1, xmmDictTypeSize);
            uni_vpaddd(xmmAux1, xmmAux1, xmmAxDimSum);
            vinsertps(dst, dst, xmmAux1, i << 4);
            add(regIdxIter, sizeof(int));
        }
        L(lExit);
    }

    void fillIndicies(Xbyak::Ymm& dst) {
        Xbyak::Label lPerXmm, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlenYmm);
        jg(lPerXmm, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, vmmDictTypeSize);
            uni_vpaddd(dst, dst, vmmAxDimSum);
            add(regIdxIter, vlenYmm);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);
        L(lPerXmm);
            for (int i = 0; i < 2; i++) {
                fillIndicies(xmmAux0);
                vinsertf128(dst, dst, xmmAux0, i);
            }
        L(lExit);
    }

    void fillIndicies(Xbyak::Zmm& dst) {
        Xbyak::Label lPerYmm, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlen);
        jg(lPerYmm, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, vmmDictTypeSize);
            uni_vpaddd(dst, dst, vmmAxDimSum);
            add(regIdxIter, vlen);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);
        L(lPerYmm);
            for (int i = 0; i < 2; i++) {
                fillIndicies(ymmAux2);
                vinsertf32x8(dst, dst, ymmAux2, i);
            }
        L(lExit);
    }

protected:
    using Vmm = typename mkldnn::impl::utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const uint32_t vlen = x64::cpu_isa_traits<isa>::vlen;
    const uint32_t vlenXmm = x64::cpu_isa_traits<x64::sse41>::vlen;
    const uint32_t vlenYmm = x64::cpu_isa_traits<x64::avx2>::vlen;
    int elPerVec;

    void gatherAndShift(const Vmm& dst, const Vmm& vmmShufMask) {
        fillIndicies(vmmSrcShifts);
        uni_vpcmpeqd(vmmOnesBit, vmmOnesBit, vmmOnesBit);
        vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmOnesBit);
        vpshufb(dst, dst, vmmShufMask);
    }

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regWorkAmount = r12;
    Xbyak::Reg64 regAux1 = r13;
    Xbyak::Reg64 regAux2 = r14;
    Xbyak::Reg64 regAux3 = r15;

    Xbyak::Reg64 regParams = x64::abi_param1;

    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(0);
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(1);
    Xbyak::Xmm xmmAxDimSum = Xbyak::Xmm(2);
    Xbyak::Xmm xmmAxDim = Xbyak::Xmm(3);
    Xbyak::Xmm xmmDictTypeSize = Xbyak::Xmm(4);

    Xbyak::Ymm ymmAux2 = Xbyak::Ymm(7);

    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
    Vmm vmmAxDimSum = Vmm(2);
    Vmm vmmAxDim = Vmm(3);
    Vmm vmmDictTypeSize = Vmm(4);
    Vmm vmmSrcShifts = Vmm(5);
    Vmm vmmOnesBit = Vmm(6);
//    Vmm vmmAux2 = Vmm(7);
    Vmm vmmAux3 = Vmm(8);
    Vmm vmmAux4 = Vmm(9);
    Vmm vmmAux5 = Vmm(10);
    Vmm vmmAux6 = Vmm(11);
    Vmm vmmAux7 = Vmm(12);
    Vmm vmmAux8 = Vmm(13);
    Vmm vmmAux9 = Vmm(14);
    Vmm vmmDst = Vmm(15);
};

class GatherImpl: public ExtLayerBase {
public:
    explicit GatherImpl(const CNNLayer* layer) {
        std::string errPrefix = std::string("Gather layer with name '") + layer->name + "' ";
        if (layer->insData.size() < 2 || layer->insData.size() > 3)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect number of input edges: " << layer->insData.size();
        if (layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect number of output edges: " << layer->outData.size();

        auto dictData = layer->insData[GATHER_DICTIONARY].lock();
        auto idxData = layer->insData[GATHER_INDEXES].lock();
        if (!dictData || !idxData)
            THROW_IE_EXCEPTION << errPrefix << "has nullable input data.";

        if (layer->insData.size() > GATHER_AXIS) {
            // TODO: implemnt when will be available via ngraph node
//            layer->insData[GATHER_AXIS]->getCreatorLayer()->blobs.begin()->second->cbuffer().as<const index_t *>() +
//            indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        } else {
            axis_ = layer->GetParamAsInt("axis");
        }

        const SizeVector& dictionaryDims = dictData->getTensorDesc().getDims();
        if (dictionaryDims.size() == 0)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect input dictionary dimension.";
        // Dictionary must be at least rank axis + 1
        IE_ASSERT(-static_cast<int>(dictionaryDims.size()) <= axis_ && axis_ < static_cast<int>(dictionaryDims.size()))
            << errPrefix << "has incorrect axis value!";
        if (axis_ < 0)
            axis_ += dictionaryDims.size();

        //  Find number of dictionaries, index range and data length
        for (int i = 0; i < axis_; i++)
            beforeAxisSize_ *= dictionaryDims[i];
        axisDim_ = dictionaryDims[axis_];
        afterAxisSize_ = 1lu;
        for (size_t i = axis_ + 1; i < dictionaryDims.size(); i++)
            afterAxisSize_ *= dictionaryDims[i];

        if (afterAxisSize_ == 0)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect input parameters dimension.";

        const SizeVector& indexesDims = idxData->getTensorDesc().getDims();
        indicesSize_ = std::accumulate(indexesDims.begin(), indexesDims.end(), 1, std::multiplies<size_t>());

        const Precision idxPrecision = Precision::I32;
        LayerConfig config;
        DataConfig dataConfigIdx, dataConfigDct;
        const Precision dictPrecision = dictData->getTensorDesc().getPrecision();
        dataConfigDct.desc = TensorDesc(dictPrecision, dictionaryDims,
                dictData->getTensorDesc().getLayoutByDims(dictionaryDims));
        config.inConfs.push_back(dataConfigDct);
        dataConfigIdx.desc = TensorDesc(idxPrecision, indexesDims,
                idxData->getTensorDesc().getLayout());
        config.inConfs.push_back(dataConfigIdx);

        DataConfig dataConfigOut;
        const SizeVector& outDims = layer->outData[0]->getTensorDesc().getDims();
        dataConfigOut.desc = TensorDesc(dictPrecision, outDims,
                layer->outData[0]->getTensorDesc().getLayoutByDims(outDims));
        config.outConfs.push_back(dataConfigOut);
        config.dynBatchSupport = false;
        confs.push_back(config);

        dictTypeSize_ = dictPrecision.size();

        // Gather instruction is applicable just for 32 and 64 bit data and is not supported by SSE.
        if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2)) &&
                afterAxisSize_ == 1) {
            jGatherConfParams jcp;
            jcp.beforeAxisSize = beforeAxisSize_;
            jcp.indicesSize = indicesSize_ * idxPrecision.size();
            jcp.dictTypeSize = dictTypeSize_;
            if (x64::mayiuse(x64::avx512_common)) {
                jKernel_.reset(new jitUniGatherKernel<x64::avx512_common>(jcp));
            } else if (x64::mayiuse(x64::avx2)) {
                jKernel_.reset(new jitUniGatherKernel<x64::avx2>(jcp));
            }
            if (jKernel_)
                jKernel_->create_ker();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (dictTypeSize_) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                return gather<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I16>::value_type):
                return gather<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                return gather<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            default:
                std::string errMsg = std::string("Gather layer has inputData input with unsupported precision: ") +
                    inputs[GATHER_DICTIONARY]->getTensorDesc().getPrecision().name();
                errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
        }
    }

private:
    template <typename dataType>
    StatusCode gather(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) {
        auto& dictionary = inputs[GATHER_DICTIONARY];
        const int* srcIndices = inputs[GATHER_INDEXES]->cbuffer().as<const int*>() +
            inputs[GATHER_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto& output = outputs[0];
//        const int *srcIndices = indexes->cbuffer().as<const int*>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

//static unsigned c1 = 0;
//static double t1 = 0.0;
//c1++;
//auto start1 = std::chrono::steady_clock::now();

        if (afterAxisSize_ == 1) {
            const dataType* srcDictData = dictionary->cbuffer().as<const dataType *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
            dataType *dstData = output->buffer().as<dataType*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

            size_t workAmount = beforeAxisSize_ * indicesSize_;
            if (jKernel_) {
                auto threadBody = [&](const int ithr, const int nthr) {
                    size_t start(0lu), end(0lu);
                    splitter(workAmount, nthr, ithr, start, end);
                    if (start >= end)
                        return;
                    size_t basStart = 0lu, idxStart = 0lu;
                    parallel_it_init(start, basStart, beforeAxisSize_, idxStart, indicesSize_);
//                    if (ithr > 0)
//                        return;
//printf("[%d] start: %lu; end: %lu; basStart: %lu; idxStart: %lu\n", ithr, start, end, basStart, idxStart);
//int tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
//int retVal = 0;

//int imm = 1;
//imm = imm | (imm << 2) | (imm << 4) | (imm << 6);
//imm = ~((imm << 3) - imm);

                    const int dictTypeSize = dictTypeSize_;
                    const int axisDimB = axisDim_ * dictTypeSize_;
                    const int axDimSumB = axisDimB * basStart;

                    auto arg = jGatherArgs();
                    arg.src = srcDictData;
                    arg.dst = dstData + basStart * indicesSize_ + idxStart;
                    arg.indices = srcIndices;
                    arg.dictTypeSize = &dictTypeSize;
                    arg.axisDim = &axisDimB;
                    arg.axDimSum = &axDimSumB;
                    arg.idxStartB = idxStart * sizeof(int);
                    arg.shufMask8bit = shufMask8bit_;
                    arg.permMask8bit = permMask8bit_;
                    arg.shufMask16bit = shufMask16bit_;
                    arg.permMask16bit = permMask16bit_;
                    arg.workAmount = end - start;
//                    arg.tmp = tmp;
//                    arg.retVal = &retVal;
                    (*jKernel_)(&arg);
//    std::string tmpStr = "tmp: ";
//for (int s = 0; s < 8; s++) {
//    tmpStr += std::to_string(tmp[s]) + "; ";
//}
//printf("%s\n", tmpStr.c_str());
//printf("retVal: %d\n", retVal);
                };

                parallel_nt(0, threadBody);
            } else {
                auto threadBody = [&](const int ithr, const int nthr) {
                    size_t start(0lu), end(0lu);
                    splitter(workAmount, nthr, ithr, start, end);
                    if (start >= end)
                        return;
                    size_t basStart = 0lu, idxStart = 0lu;
                    parallel_it_init(start, basStart, beforeAxisSize_, idxStart, indicesSize_);

                    for (size_t i = basStart; i < beforeAxisSize_ && start < end; i++) {
                        const dataType* srcDictDataShifted = srcDictData + i * axisDim_;
                        dataType* dstDataShifted = dstData + i * indicesSize_;
                        for (size_t j = idxStart; j < indicesSize_ && start < end; j++, start++) {
                            dstDataShifted[j] = srcDictDataShifted[srcIndices[j]];
                        }
                        idxStart = 0lu;
                    }
                };

                parallel_nt(0, threadBody);
            }
        } else {
            const uint8_t *srcDictData = dictionary->cbuffer().as<const uint8_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
            uint8_t* dstData = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
            const size_t len = afterAxisSize_ * dictionary->getTensorDesc().getPrecision().size();
            const size_t idxMlt = len * axisDim_;
            const size_t lenSrcIndexSize = len * indicesSize_;
            parallel_for(indicesSize_, [&](size_t i) {
                //  Index clipping
                size_t len_i = len * i;
                size_t dstSize = output->byteSize() - len_i;
                if (srcIndices[i] < axisDim_) {
                    size_t idxShift = len * srcIndices[i];
                    uint8_t* dstDataShifted = dstData + len_i;
                    const uint8_t* srcDictDataShifted = srcDictData + idxShift;
                    for (size_t j = 0; j < beforeAxisSize_; j++) {
                        size_t jlenSrcIndexSize = j * lenSrcIndexSize;
                        cpu_memcpy_s(dstDataShifted + jlenSrcIndexSize,
                                    dstSize - jlenSrcIndexSize,
                                    srcDictDataShifted + j * idxMlt,
                                    len);
                    }
                } else {
                    for (size_t j = 0; j < beforeAxisSize_; j++) {
                        memset(&dstData[len * (i + j * indicesSize_)], 0, len);
                    }
                }
            });
        }

//auto end1 = std::chrono::steady_clock::now();
//t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
//if (c1 % 1000 == 0) {
//    std::cout << "GE PARALLEL SECTION: " << t1 / c1 << std::endl;
//}

        return OK;
    }

    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
    const size_t GATHER_AXIS = 2;

    int axis_ = 0;
    size_t beforeAxisSize_ = 1lu;
    size_t axisDim_ = 0lu;
    size_t afterAxisSize_ = 1lu;
    size_t indicesSize_ = 1lu;
    size_t dictTypeSize_ = 1lu;
    int shufMask8bit_[8] = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
    int permMask8bit_[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    int shufMask16bit_[8] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
    int permMask16bit_[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    std::shared_ptr<jitGatherKernelBase> jKernel_;
};


REG_FACTORY_FOR(GatherImpl, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
