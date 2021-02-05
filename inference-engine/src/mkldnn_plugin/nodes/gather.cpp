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
//    uint32_t indicesSize;
};

struct jGatherArgs {
    const void* src;
    void* dst;
    const int* indices;
    const int* dictTypeSize;
    const int* axisDim;
    const int* axDimSum;
    size_t idxStart;
    size_t workAmount;
    int* tmp;
    int* retVal;
};

struct jitUniGatherKernel {
    void (*ker_)(const jGatherArgs *);
    void operator()(const jGatherArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitUniGatherKernel(jGatherConfParams jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jitUniGatherKernel() {}

    virtual void create_ker() = 0;

    jGatherConfParams jcp_;
};

#define GET_OFF(field) offsetof(jGatherArgs, field)

template <x64::cpu_isa_t isa>
struct jitUniGatherKernel_32 : public jitUniGatherKernel, public x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel_32)

    explicit jitUniGatherKernel_32(jGatherConfParams jcp) : jitUniGatherKernel(jcp), x64::jit_generator() {}

    void create_ker() override {
        x64::jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(regSrc, ptr[regParams + GET_OFF(src)]);
        mov(regDst, ptr[regParams + GET_OFF(dst)]);
        mov(regIndices, ptr[regParams + GET_OFF(indices)]);

//        mov(regIdxShifted, regIndices);
        mov(regIdxIter, ptr[regParams + GET_OFF(idxStart)]);
//        add(regIdxShifted, regIdxIter);

        mov(regTmp, ptr[regParams + GET_OFF(dictTypeSize)]);
        uni_vpbroadcastd(vmmDictTypeSize, ptr[regTmp]);

        mov(regAxisDim, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxDim, ptr[regAxisDim]);

        mov(regTmp, ptr[regParams + GET_OFF(axDimSum)]);
        uni_vpbroadcastd(vmmAxDimSum, ptr[regTmp]);

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

        mov(regTmp, ptr[regParams + GET_OFF(tmp)]);
//        mov(regTmp, ptr[regParams + GET_OFF(retVal)]);

        elPerVec = vlen / jcp_.dictTypeSize;

        uni_vxorps(vmmZero, vmmZero, vmmZero);

        if (jcp_.dictTypeSize == 4) {
            Xbyak::Label lDstIdxLoop, lTail, lFinish;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                fillIndicies(vmmSrcIdx);

                uni_vpcmpeqd(vmmOnesBit, vmmOnesBit, vmmOnesBit);
                vpgatherdd(vmmDst, ptr[regSrc + vmmSrcIdx], vmmOnesBit);
                uni_vmovups(ptr[regDst], vmmDst);
//                uni_vmovups(ptr[regTmp], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            {
                cmp(regWorkAmount, 0);
                je(lFinish, T_NEAR);

                cmp(regIdxIter, jcp_.indicesSize);
                jl(insertLabel, T_NEAR);

//                mov(regTmp, ptr[regIndices]);
//                sub(regTmp, regIdxIter);
//                mov(rax, regTmp);
//                mul(regRetVal);
//                vextractps(regTmp32, Xbyak::Xmm(vmmDstShift0.getIdx()), 0);
//                add(regTmp, eax);
//                mov(rdx, ptr[regSrc]);
//                mov(ptr[regDst], rdx);

//                add(regSrc, jcp.dictTypeSize);
//                add(regDst, jcp.dictTypeSize);
//                add(regIndices, sizeof(int));
                sub(regWorkAmount, 1);
                jmp(lTail, T_NEAR);
            }
//
////            L(strideFinish);
////            mov(regIdxIter, 0);
////            uni_vpaddd(vmmDictTypeSize, vmmDictTypeSize, vmmOnes);
////            inc(regIdxShifted);
////            cmp(regIdxShifted, jcp.dstAxDim);
////            jl(lDstIdxLoop, T_NEAR);
////            mov(regIdxShifted, 0);
////            uni_movaps(vmmDictTypeSize, vmmZero);
////            uni_vpaddd(vmmDstShift0, vmmDstShift0, vmmStrideAx1Diff);
////
////            jmp(lDstIdxLoop, T_NEAR);
////        }
            L(lFinish);
        } else if (jcp_.dictTypeSize == 2) {
        } else if (jcp_.dictTypeSize == 1) {
        }

        this->postamble();
    }

    inline void uni_insertps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op, uint8_t imm) {
        if (isa == x64::avx512_common || isa == x64::avx2) {
            vinsertps(x1, x2, op, imm);
        } else {
            insertps(x1, op, imm);
        }
    }

    inline void uni_movaps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2) {
        if (isa == x64::avx512_common || isa == x64::avx2) {
            vmovaps(x1, x2);
        } else {
            movaps(x1, x2);
        }
    }

    inline void uni_vpmuldq(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& x3) {
        if (isa == x64::avx512_common || isa == x64::avx2) {
            vpmuldq(x1, x2, x3);
        } else {
            pmuldq(x1, x3);
        }
    }

    void fillIndicies(Xbyak::Xmm& dst) {
        uni_vmovups(vmmAux1, ptr[regIndices + regIdxIter]);
        uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
        uni_vpaddd(vmmAux1, vmmAux1, vmmAxDimSum);
        for (int i = 0; i < 4; i++) {
            Xbyak::Label insertLabel, incLabel;

            cmp(regIdxIter, jcp_.indicesSize);
            jl(insertLabel, T_NEAR);
            mov(regIdxIter, 0);
            uni_vmovups(vmmAux1, ptr[regIndices]);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            uni_vpmulld(vmmAux1, vmmAux1, vmmDictTypeSize);
            uni_vpaddd(vmmAux1, vmmAux1, vmmAxDimSum);

            L(insertLabel);
            uni_insertps(dst, xmmAux1, xmmAux1, i << 6);
            add(regIdxIter, sizeof(int));
        }
    }

    void fillIndicies(Xbyak::Ymm& dst) {
        Xbyak::Label lPerElement, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlen);
        jg(lPerElement, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, vmmDictTypeSize);
            uni_vpaddd(dst, dst, vmmAxDimSum); //check +*
            add(regIdxIter, vlen);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);
        L(lPerElement);
            for (int i = 0; i < 2; i++) {
                fillIndicies(xmmAux0);
                vinsertf128(dst, dst, xmmAux0, i);
            }
        L(lExit);
    }

    void fillIndicies(Xbyak::Zmm& dst) {
        for (int i = 0; i < 2; i++) {
            fillIndicies(ymmDstAxIdx);
            vinsertf32x8(dst, dst, ymmDstAxIdx, i);
        }
    }

private:
    using Vmm = typename mkldnn::impl::utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = x64::cpu_isa_traits<isa>::vlen;
    int elPerVec;

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regIdxShifted = r12;
    Xbyak::Reg64 regAxisDim = r13;
    Xbyak::Reg64 regWorkAmount = r14;
    Xbyak::Reg64 regTmp = r15;
    Xbyak::Reg32 regTmp32 = r15d;

    Xbyak::Reg64 regParams = x64::abi_param1;

    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(0);
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(1);
    Xbyak::Ymm ymmDstAxIdx = Xbyak::Ymm(0);
//    Xbyak::Ymm ymmDstShift0 = Xbyak::Ymm(1);
    Vmm vmmDictTypeSize = Vmm(8);
    Vmm vmmDstShift0 = Vmm(9);

//    Xbyak::Xmm xmmDstAxIdxAux = Xbyak::Xmm(2);
    Xbyak::Xmm xmmAxDimSum = Xbyak::Xmm(3);
    Xbyak::Xmm xmmAxDim = Xbyak::Xmm(4);
//    Xbyak::Xmm xmmZero = Xbyak::Xmm(5);
//    Xbyak::Xmm xmmOnes = Xbyak::Xmm(6);

    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
    Vmm vmmAxDimSum = Vmm(3);
    Vmm vmmAxDim = Vmm(4);
    Vmm vmmZero = Vmm(5);
    Vmm vmmOnes = Vmm(6);
    Vmm vmmIncVec = Vmm(7);
    Vmm vmmIndicies = Vmm(10);
    Vmm vmmSrcIdx = Vmm(11);
    Vmm vmmStrideAx1Diff = Vmm(12);
    Vmm vmmStrideAxSrc = Vmm(13);
    Vmm vmmOnesBit = Vmm(14);
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
        if (dictTypeSize_ == sizeof(PrecisionTrait<Precision::I32>::value_type) &&
                (x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2)) &&
                afterAxisSize_ == 1) {
            jGatherConfParams jcp;
            jcp.beforeAxisSize = beforeAxisSize_;
            jcp.indicesSize = indicesSize_ * idxPrecision.size();
            jcp.dictTypeSize = dictTypeSize_;
            if (x64::mayiuse(x64::avx512_common)) {
                kernel32_.reset(new jitUniGatherKernel_32<x64::avx512_common>(jcp));
            } else if (x64::mayiuse(x64::avx2)) {
                kernel32_.reset(new jitUniGatherKernel_32<x64::avx2>(jcp));
            }
            if (kernel32_)
                kernel32_->create_ker();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
//        switch (inputs[GATHER_INDEXES]->getTensorDesc().getPrecision()) {
//            case Precision::FP32:
//                gather<float, f32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
//                break;
////            case Precision::FP16:
////                gather<ie_fp16, f16toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
////                break;
//            case Precision::I32:
//                gather<int32_t, i32toUi32>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                switch (dictTypeSize_) {
                    case sizeof(PrecisionTrait<Precision::I32>::value_type):
                        gather<PrecisionTrait<Precision::I32>::value_type>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                        break;
                    case sizeof(PrecisionTrait<Precision::I16>::value_type):
                        gather<PrecisionTrait<Precision::I16>::value_type>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                        break;
                    case sizeof(PrecisionTrait<Precision::I8>::value_type):
                        gather<PrecisionTrait<Precision::I8>::value_type>(inputs[GATHER_INDEXES], inputs[GATHER_DICTIONARY], outputs[0]);
                        break;
                }
//                break;
//            default:
//                return GENERAL_ERROR;
//        }

        return OK;
    }

private:
//    template <typename index_t, class Conversion>
//    void gather(Blob::Ptr indexes, Blob::Ptr dictionary, Blob::Ptr output) {
//        size_t indicesSize_ = indexes->size();
//        const index_t *srcIndices = indexes->cbuffer().as<const index_t *>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
//        const uint8_t *srcDictData = dictionary->cbuffer().as<const uint8_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
//        uint8_t *dstData = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
//        size_t len = afterAxisSize_ * dictionary->getTensorDesc().getPrecision().size();
//
//static unsigned c1 = 0;
//static double t1 = 0.0;
//c1++;
//auto start1 = std::chrono::steady_clock::now();
//
//        parallel_for(indicesSize_, [&](size_t i) {
//            unsigned int idx = Conversion()(srcIndices[i]);
//
//            //  Index clipping
//            if (idx < axisDim_) {
//                //  Copying data to destination from Dictionary
//                for (size_t j = 0; j < beforeAxisSize_; j++) {
//                    cpu_memcpy_s(&dstData[len * (i + j * indicesSize_)],
//                                output->byteSize() - (len * (i + j * indicesSize_)),
//                                &srcDictData[len * (idx + j * axisDim_)],
//                                len);
//                }
//            } else {
//                for (size_t j = 0; j < beforeAxisSize_; j++) {
//                    memset(&dstData[len * (i + j * indicesSize_)], 0, len);
//                }
//            }
//        });
//
//auto end1 = std::chrono::steady_clock::now();
//t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
//if (c1 % 1000 == 0) {
//    std::cout << "GE PARALLEL SECTION: " << t1 / c1 << std::endl;
//}
//    }

    template <typename dataType>
    void gather(Blob::Ptr& indexes, Blob::Ptr& dictionary, Blob::Ptr& output) {
        const int *srcIndices = indexes->cbuffer().as<const int*>() + indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();

static unsigned c1 = 0;
static double t1 = 0.0;
c1++;
auto start1 = std::chrono::steady_clock::now();

        if (afterAxisSize_ == 1) {
            const dataType* srcDictData = dictionary->cbuffer().as<const dataType *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
            dataType *dstData = output->buffer().as<dataType*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

//            parallel_for(indicesSize_, [&](size_t i) {
//                //  Index clipping
//                if (srcIndices[i] < axisDim_) {
//                    //  Copying data to destination from Dictionary
//                    for (size_t j = 0; j < beforeAxisSize_; j++) {
//                        dstData[i + j * indicesSize_] = srcDictData[srcIndices[i] + j * axisDim_];
//                    }
//                } else {
//                    for (size_t j = 0; j < beforeAxisSize_; j++) {
////                        memset(&dstData[len * (i + j * indicesSize_)], 0, len);
//                        dstData[i + j * indicesSize_] = 0;
//                    }
//                }
//            });
//            parallel_for(beforeAxisSize_, [&](size_t i) {
//                const dataType* srcDictDataShifted = srcDictData + i * axisDim_;
//                dataType* dstDataShifted = dstData + i * indicesSize_;
//                    for (size_t j = 0; j < indicesSize_; j++) {
//                        dstDataShifted[j] = srcDictDataShifted[srcIndices[j]];
//                    }
//            });
//            parallel_for2d(beforeAxisSize_, indicesSize_, [&](size_t i, size_t j) {
//                dstData[j + i * indicesSize_] = srcDictData[srcIndices[j] + i * axisDim_];
//            });

            size_t workAmount = beforeAxisSize_ * indicesSize_;
            if (kernel32_) {
                auto threadBody = [&](const int ithr, const int nthr) {
                    size_t start(0lu), end(0lu);
                    splitter(workAmount, nthr, ithr, start, end);
                    if (start >= end)
                        return;
                    size_t basStart = 0lu, idxStart = 0lu;
                    parallel_it_init(start, basStart, beforeAxisSize_, idxStart, indicesSize_);
//                    if (ithr > 0)
//                        return;
printf("[%d] start: %lu; end: %lu; basStart: %lu; idxStart: %lu\n", ithr, start, end, basStart, idxStart);
int tmp[8];
int retVal = 0;
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
                    arg.idxStart = idxStart * dictTypeSize_;
                    arg.workAmount = end - start;
                    arg.tmp = tmp;
                    arg.retVal = &retVal;
                    (*kernel32_)(&arg);
    std::string tmpStr = "tmp: ";
for (int s = 0; s < 8; s++) {
    tmpStr += std::to_string(tmp[s]) + "; ";
}
printf("%s\n", tmpStr.c_str());
printf("retVal: %d\n", retVal);
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

auto end1 = std::chrono::steady_clock::now();
t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
if (c1 % 1000 == 0) {
    std::cout << "GE PARALLEL SECTION: " << t1 / c1 << std::endl;
}
    }

    int axis_ = 0;
    size_t beforeAxisSize_ = 1lu;
    size_t axisDim_ = 0lu;
    size_t afterAxisSize_ = 1lu;
    size_t indicesSize_ = 1lu;
    size_t dictTypeSize_ = 1lu;
    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
    const size_t GATHER_AXIS = 2;
    std::shared_ptr<jitUniGatherKernel> kernel32_;
};


REG_FACTORY_FOR(GatherImpl, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
