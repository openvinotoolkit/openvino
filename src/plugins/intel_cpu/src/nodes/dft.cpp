// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include <common/primitive_hashing.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ngraph/opsets/opset7.hpp>

#include "common/cpu_memcpy.h"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include "utils/general_utils.h"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace InferenceEngine;
using namespace Xbyak;

#define GET_OFF_DFT(field) offsetof(jit_args_dft, field)
#define GET_OFF_FFT(field) offsetof(jit_args_fft, field)

namespace {
struct DFTKey {
    ov::intel_cpu::node::DFT::DFTAttrs nodeAttrs;

    size_t hash() const;
    bool operator==(const DFTKey& rhs) const;
};

size_t DFTKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, nodeAttrs.inverse);

    return seed;
}

bool DFTKey::operator==(const DFTKey& rhs) const {
    if (nodeAttrs.inverse != rhs.nodeAttrs.inverse)
        return false;

    return true;
}

}  // namespace

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_dft_config_params {
    bool inverse;
};

template <cpu_isa_t isa>
struct jit_uni_dft_kernel_f32 : public jit_uni_dft_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dft_kernel_f32)

    jit_uni_dft_kernel_f32(jit_dft_config_params jcp) : jcp_(jcp), jit_uni_dft_kernel(), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF_DFT(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF_DFT(dst)]);
        mov(reg_twiddles, ptr[reg_params + GET_OFF_DFT(twiddles)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF_DFT(work_amount)]);
        mov(reg_index, ptr[reg_params + GET_OFF_DFT(index)]);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        mov(aux_reg_work_amount, reg_work_amount);
        uni_vpxor(vmm_sum, vmm_sum, vmm_sum);

        int step = vlen / 8;

        L(main_loop_label);
        {
            cmp(aux_reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            uni_vpshufd(vmm_data, ptr[reg_src], 0b01000001);
            uni_vpshufd(vmm_twiddles, ptr[reg_twiddles], 0b01000100);
            uni_vfmadd231ps(vmm_sum, vmm_data, vmm_twiddles);

            uni_vpshufd(vmm_data, ptr[reg_src], 0b11101011);
            uni_vpshufd(vmm_twiddles, ptr[reg_twiddles], 0b11101110);
            uni_vfmadd231ps(vmm_sum, vmm_data, vmm_twiddles);

            add(reg_twiddles, 2 * step * sizeof(float));
            add(reg_src, 2 * step * sizeof(float));

            sub(aux_reg_work_amount, step);
            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        if (mayiuse(cpu::x64::avx512_core)) {
            Xbyak::Zmm zmm_sum = Xbyak::Zmm(vmm_sum.getIdx());
            Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_sum.getIdx());
            Xbyak::Ymm ymm_sum_2 = Xbyak::Ymm(vmm_sum_2.getIdx());

            vextractf64x4(ymm_sum_2, zmm_sum, 1);
            vaddps(ymm_sum, ymm_sum, ymm_sum_2);
        }
        if (mayiuse(cpu::x64::avx2)) {
            Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_sum.getIdx());

            vextractf128(xmm_sum_2, ymm_sum, 1);
            vaddps(xmm_sum, xmm_sum, xmm_sum_2);
        }

        L(tail_loop_label);
        {
            cmp(aux_reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            uni_vpshufd(xmm_data, ptr[reg_src], 0b01000001);
            uni_vpshufd(xmm_twiddles, ptr[reg_twiddles], 0b01000100);
            uni_vfmadd231ps(xmm_sum, xmm_data, xmm_twiddles);

            add(reg_twiddles, 2 * sizeof(float));
            add(reg_src, 2 * sizeof(float));

            sub(aux_reg_work_amount, 1);
            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);

        vmovhlps(xmm_sum_2, xmm_sum_2, xmm_sum);

        if (!jcp_.inverse) {
            vhsubps(xmm_sum_2, xmm_sum_2, xmm_sum_2);
            vhaddps(xmm_sum, xmm_sum, xmm_sum);
        } else {
            vhaddps(xmm_sum_2, xmm_sum_2, xmm_sum_2);
            vhsubps(xmm_sum, xmm_sum, xmm_sum);

            uni_vmovq(xmm_div, reg_work_amount);
            uni_vcvtdq2ps(xmm_div, xmm_div);
            vdivss(xmm_sum, xmm_sum, xmm_div);
            vdivss(xmm_sum_2, xmm_sum_2, xmm_div);
        }

        uni_vmovss(ptr[reg_dst], xmm_sum_2);
        uni_vmovss(ptr[reg_dst + sizeof(float)], xmm_sum);

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_twiddles = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 aux_reg_work_amount = r12;
    Xbyak::Reg64 reg_index = r13;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_data = Vmm(0);
    Xmm xmm_data = Xmm(0);
    Vmm vmm_twiddles = Vmm(1);
    Xmm xmm_twiddles = Xmm(1);
    Vmm vmm_sum = Vmm(2);
    Xmm xmm_sum = Xmm(2);

    Vmm vmm_sum_2 = vmm_data;
    Xmm xmm_sum_2 = xmm_data;
    Xmm xmm_div = xmm_twiddles;

    jit_dft_config_params jcp_ = {};
};

template <cpu_isa_t isa>
struct jit_uni_fft_kernel_f32 : public jit_uni_fft_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_fft_kernel_f32)

    jit_uni_fft_kernel_f32(jit_dft_config_params jcp) : jcp_(jcp), jit_uni_fft_kernel(), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    template<typename T>
    void loop_process(int step) {
        T reg_data_odd_1 = T(vmm_data_odd_1.getIdx());
        T reg_data_odd_2 = T(vmm_data_odd_2.getIdx());
        T reg_twiddle_imag = T(vmm_twiddle_imag.getIdx());
        T reg_twiddle_real = T(vmm_twiddle_real.getIdx());
        T reg_data_even = T(vmm_data_even.getIdx());
        T reg_data_result = T(vmm_data_result.getIdx());
        T reg_negative_mask = T(vmm_negative_mask.getIdx());

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;

        L(loop_label);
        {
            cmp(aux_reg_work_amount, step);
            jl(loop_end_label, T_NEAR);

            move_data(reg_data_odd_1, ptr[reg_src + reg_even_in_diff], step);
            uni_vpshufd(reg_data_odd_2, reg_data_odd_1, 0b10110001);
            uni_vmulps(reg_data_odd_2, reg_data_odd_2, reg_twiddle_imag);

            if (mayiuse(cpu::x64::avx512_core)) {
                if (!jcp_.inverse) {
                    vfmaddsub213ps(reg_data_odd_1, reg_twiddle_real, reg_data_odd_2);
                } else {
                    vfmsubadd213ps(reg_data_odd_1, reg_twiddle_real, reg_data_odd_2);
                }
            } else {
                if (jcp_.inverse) {
                    uni_vxorps(reg_data_odd_2, reg_data_odd_2, reg_negative_mask);
                }

                uni_vmulps(reg_data_odd_1, reg_data_odd_1, reg_twiddle_real);
                vaddsubps(reg_data_odd_1, reg_data_odd_1, reg_data_odd_2);
            }

            move_data(reg_data_even, ptr[reg_src], step);

            uni_vaddps(reg_data_result, reg_data_even, reg_data_odd_1);
            move_data(ptr[reg_dst], reg_data_result, step);

            uni_vsubps(reg_data_result, reg_data_even, reg_data_odd_1);
            move_data(ptr[reg_dst + reg_even_out_diff], reg_data_result, step);

            add(reg_src, step * sizeof(float));
            add(reg_dst, step * sizeof(float));

            sub(aux_reg_work_amount, step);
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF_FFT(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF_FFT(dst)]);
        mov(reg_twiddles_addr, ptr[reg_params + GET_OFF_FFT(twiddles)]);

        mov(reg_num_blocks, ptr[reg_params + GET_OFF_FFT(num_blocks)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF_FFT(work_amount)]);

        mov(reg_even_in_diff, sizeof(float));
        mul(ptr[reg_params + GET_OFF_FFT(n_complex)]);
        mov(reg_even_out_diff, reg_even_in_diff);

        mov(reg_even_in_diff, sizeof(float));
        mul(reg_work_amount);

        if (jcp_.inverse) {
            Xbyak::Reg32 reg_negative_mask = Xbyak::Reg32(aux_reg_work_amount.getIdx());
            Xbyak::Xmm xmm_negative_mask = Xbyak::Xmm(vmm_negative_mask.getIdx());

            mov(reg_negative_mask, 0x80000000);
            uni_vmovd(xmm_negative_mask, reg_negative_mask);
            uni_vpbroadcastd(vmm_negative_mask, xmm_negative_mask);
        }

        Xbyak::Label block_loop_label;
        Xbyak::Label block_loop_end_label;

        L(block_loop_label);
        {
            cmp(reg_num_blocks, 1);
            jl(block_loop_end_label, T_NEAR);

            mov(aux_reg_work_amount, reg_work_amount);
            uni_vbroadcastss(vmm_twiddle_real, ptr[reg_twiddles_addr]);
            uni_vbroadcastss(vmm_twiddle_imag, ptr[reg_twiddles_addr + sizeof(float)]);

            if (mayiuse(cpu::x64::avx2)) {
                loop_process<Vmm>(vlen / 4);
            }
            loop_process<Xmm>(4);
            loop_process<Xmm>(2);

            add(reg_twiddles_addr, 2 * sizeof(float));
            add(reg_src, reg_even_in_diff);
            sub(reg_num_blocks, 1);

            jmp(block_loop_label, T_NEAR);
        }
        L(block_loop_end_label);

        this->postamble();
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg32 aux_negative_mask = r8d;

    Xbyak::Reg64 reg_even_in_diff = rax;
    Xbyak::Reg64 reg_even_out_diff = rbx;

    Xbyak::Reg64 reg_src = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_num_blocks = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 aux_reg_work_amount = r13;
    Xbyak::Reg64 reg_twiddles_addr = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_data_odd_1 = Vmm(0);
    Vmm vmm_data_odd_2 = Vmm(1);
    Vmm vmm_twiddle_real = Vmm(2);
    Vmm vmm_twiddle_imag = Vmm(3);
    Vmm vmm_negative_mask = Vmm(4);
    Vmm vmm_data_even = Vmm(5);

    Vmm vmm_data_result = vmm_data_odd_2;

    jit_dft_config_params jcp_ = {};

    inline void move_data(const Xbyak::Address& addr, const Xmm& x, int count) {
        if (count == 2) {
            uni_vmovq(addr, x);
        } else {
            uni_vmovups(addr, x);
        }
    }

    inline void move_data(const Xmm& x, const Xbyak::Address& addr, int count) {
        if (count == 2) {
            uni_vmovq(x, addr);
        } else {
            uni_vmovups(x, addr);
        }
    }
};


bool DFT::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto interpDFT = std::dynamic_pointer_cast<const ngraph::opset7::DFT>(op);
        const auto interpIDFT = std::dynamic_pointer_cast<const ngraph::opset7::IDFT>(op);

        if (!interpDFT && !interpIDFT) {
            errorMessage = "Only opset7 DFT/IDFT operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

DFT::DFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr& cache)
    : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    layerErrorPrefix = "DFT layer with name '" + op->get_name() + "'";
    const size_t inputsNumber = getOriginalInputsNumber();
    if (inputsNumber != 2 && inputsNumber != 3) {
        IE_THROW() << layerErrorPrefix << " has invalid number of input/output edges: " << inputsNumber;
    }

    /* Data */
    inputShape = inputShapes[DATA_INDEX].getStaticDims();
    if (inputShape.size() < 2) {
        IE_THROW() << layerErrorPrefix << " has invalid 'data' input tensor with rank: " << inputShape.size();
    }

    /* Axes */
    const auto axesRank = inputShapes[AXES_INDEX].getRank();
    if (axesRank != 1) {
        IE_THROW() << layerErrorPrefix << " has invalid 'axes' input tensor with rank: " << axesRank;
    }

    /* Signal size */
    if (inputsNumber > SIGNAL_SIZE_INDEX) {
        const auto signalSizeRank = inputShapes[SIGNAL_SIZE_INDEX].getRank();
        if (signalSizeRank != 1) {
            IE_THROW() << layerErrorPrefix << " has invalid 'signal_size' input tensor with rank: " << signalSizeRank;
        }
    }

    inverse = std::dynamic_pointer_cast<ngraph::opset7::DFT>(op) == nullptr;
}

void DFT::getSupportedDescriptors() {}

void DFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!dataPrecision.is_float()) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();
    }

    const auto& axesPrecision = getOriginalInputPrecisionAtPort(AXES_INDEX);
    if (axesPrecision != Precision::I32 && axesPrecision != Precision::I64) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesPrecision.name();
    }

    if (inputShapes.size() > SIGNAL_SIZE_INDEX) {
        const auto& signalSizeTensorPrec = getOriginalInputPrecisionAtPort(SIGNAL_SIZE_INDEX);
        if (signalSizeTensorPrec != Precision::I32 && signalSizeTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix
                       << " has unsupported 'signal_size' input precision: " << signalSizeTensorPrec.name();
        }
    }

    std::vector<PortConfigurator> inDataConfigurators(
        {{LayoutType::ncsp, Precision::FP32}, {LayoutType::ncsp, Precision::I32}});
    if (inputShapes.size() > SIGNAL_SIZE_INDEX)
        inDataConfigurators.push_back({LayoutType::ncsp, Precision::I32});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, Precision::FP32}}, impl_desc_type::ref_any);
}

namespace {
inline float getRealFromComplexProd(float lhsReal, float lhsImag, float rhsReal, float rhsImag) {
    return lhsReal * rhsReal - lhsImag * rhsImag;
}

inline float getImaginaryFromComplexProd(float lhsReal, float lhsImag, float rhsReal, float rhsImag) {
    return lhsReal * rhsImag + lhsImag * rhsReal;
}

/*
    Returns true while we can iterate
    Specified axis is skipped in counters
*/
inline bool nextIterationStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange, size_t axis) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();

    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        if (std::distance(itCounter, counters.rend()) == axis + 1) {
            ++itCounter;
            ++itWork;
            continue;
        }
        *itCounter = (*itCounter + 1) % *itWork;
        if (*itCounter != 0) {
            return true;
        }
        ++itCounter;
        ++itWork;
    }
    return false;
}

inline bool IsPowerOfTwo(size_t n) {
    return (n != 0) && (n & (n - 1)) == 0;
}

inline bool copyStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();

    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        *itCounter = (*itCounter + 1) % *itWork;
        if (*itCounter != 0) {
            return true;
        }
        ++itCounter;
        ++itWork;
    }
    return false;
}

size_t calculateOffsetFromStrides(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t index = 0; index < coords.size(); ++index) {
        offset += coords[index] * strides[index];
    }
    return offset;
}

void gatherToBufferND(float* buffer,
                      const float* data,
                      size_t axis,
                      const std::vector<size_t>& dimIndexes,
                      const std::vector<size_t>& shape,
                      const std::vector<size_t>& strides) {
    size_t numberOfComplex = shape[axis];
    size_t offset = calculateOffsetFromStrides(dimIndexes, strides);

    for (size_t bufferIndex = 0; bufferIndex < 2 * numberOfComplex; bufferIndex += 2) {
        buffer[bufferIndex] = data[offset];
        buffer[bufferIndex + 1] = data[offset + 1];
        offset += strides[axis];
    }
}

void applyBufferND(const float* buffer,
                   float* output,
                   size_t axis,
                   const std::vector<size_t>& dimIndexes,
                   const std::vector<size_t>& shape,
                   const std::vector<size_t>& strides) {
    size_t numberOfComplex = shape[axis];
    size_t offset = calculateOffsetFromStrides(dimIndexes, strides);

    for (size_t bufferIndex = 0; bufferIndex < 2 * numberOfComplex; bufferIndex += 2) {
        output[offset] = buffer[bufferIndex];
        output[offset + 1] = buffer[bufferIndex + 1];
        offset += strides[axis];
    }
}

void copyDataToOutputWithSignalSize(const float* input, const std::vector<size_t>& inputShape, const std::vector<size_t>& inputStrides,
                                    float* output, const std::vector<size_t>& outputShape, const std::vector<size_t>& outputStrides) {
    auto totalInput = std::accumulate(inputShape.begin(), inputShape.end(), size_t(1), std::multiplies<size_t>());
    auto totalOutput = std::accumulate(outputShape.begin(), outputShape.end(), size_t(1), std::multiplies<size_t>());
    std::fill_n(output, totalOutput, 0.f);
    size_t lastChangedDim = 0;
    for (size_t index = inputShape.size() - 1; index > 0; --index) {
        if (inputShape[index] != outputShape[index]) {
            lastChangedDim = index;
            break;
        }
    }
    if (lastChangedDim == 0) {
        size_t outputBytesSize = std::min(totalOutput, totalInput) * sizeof(float);
        cpu_memcpy(output, input, outputBytesSize);
        return;
    }

    std::vector<size_t> iterationRange(lastChangedDim + 1, 0);
    for (size_t index = 0; index < lastChangedDim + 1; ++index) {
        iterationRange[index] = std::min(inputShape[index], outputShape[index]);
    }

    const std::vector<size_t> inputStridesRange(inputStrides.begin(), inputStrides.begin() + iterationRange.size());
    const std::vector<size_t> outputStridesRange(outputStrides.begin(), outputStrides.begin() + iterationRange.size());
    const size_t blockSize = std::accumulate(inputShape.begin() + lastChangedDim + 1, inputShape.end(), size_t(1), std::multiplies<size_t>());
    const size_t blockSizeBytes = blockSize * sizeof(float);
    std::vector<size_t> iterationCounter(iterationRange.size(), 0);
    do {
        size_t offsetInput = calculateOffsetFromStrides(iterationCounter, inputStrides);
        size_t offsetOutput = calculateOffsetFromStrides(iterationCounter, outputStrides);
        cpu_memcpy(output + offsetOutput, input + offsetInput, blockSizeBytes);
    } while (copyStep(iterationCounter, iterationRange));
}

}  // namespace

void DFT::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute DFT node with name: " << getName() << ". Primitive isn't created";
        return;
    }

    auto axesEdge = getParentEdgeAt(AXES_INDEX);
    const auto* axesStartPtr = reinterpret_cast<const int32_t*>(axesEdge->getMemoryPtr()->GetPtr());
    auto axes = std::vector<int32_t>(axesStartPtr, axesStartPtr + axesEdge->getMemory().getStaticDims()[0]);
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inputShape.size() - 1;
        }
    }
    std::sort(axes.begin(), axes.end());

    auto outputShape = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    auto inputDataEdge = getParentEdgeAt(DATA_INDEX);
    auto outputDataEdge = getChildEdgeAt(0);

    const auto src = reinterpret_cast<const float*>(inputDataEdge->getMemoryPtr()->GetPtr());
    auto dst = reinterpret_cast<float*>(outputDataEdge->getMemoryPtr()->GetPtr());

    auto inputRank = inputDataEdge->getMemory().GetShape().getRank();

    auto inputStrides = inputDataEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    auto outputStrides = outputDataEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

    execPtr->exec(src, dst, inputRank, axes, inputShape, outputShape, inputStrides, outputStrides, inverse);
}

void DFT::DFTExecutor::exec(const float* src,
                            float* dst,
                            size_t inputRank,
                            std::vector<int32_t> axes,
                            VectorDims inputShape,
                            VectorDims outputShape,
                            VectorDims inputStrides,
                            VectorDims outputStrides,
                            bool inverse) {
    size_t nComplexMaxFFT = 0;
    for (size_t axis : axes) {
        size_t nComplex = outputShape[axis];
        // FFT uses different twiddle factors
        if (!IsPowerOfTwo(nComplex)) {
            if (twiddlesMapDFT.find(nComplex) == twiddlesMapDFT.end()) {
                twiddlesMapDFT[nComplex] = generateTwiddlesDFT(nComplex);
            }
        } else {
            if (nComplexMaxFFT < nComplex) {
                nComplexMaxFFT = nComplex;
            }
        }
    }

    if (nComplexMaxFFT > 0 && (nComplexMaxFFT - 1) * 2 > twiddlesFFT.size()) {
        generateTwiddlesFFT(nComplexMaxFFT);
    }

    if (inputShape != outputShape) {
        copyDataToOutputWithSignalSize(src,
                                       inputShape,
                                       inputStrides,
                                       dst,
                                       outputShape,
                                       outputStrides);
    } else {
        auto totalElements = std::accumulate(inputShape.begin(), inputShape.end(), size_t(1), std::multiplies<size_t>());
        cpu_memcpy(dst, src, totalElements * sizeof(float));
    }

    // 1d case
    if (inputRank == 2) {
        size_t nComplex = outputShape[0];
        if (IsPowerOfTwo(nComplex)) {
            fft(dst, nComplex * 2, inverse, true);
        } else {
            naiveDFT(dst, nComplex * 2, inverse);
        }
    } else {
        dftNd(dst, outputShape, outputStrides, axes, inverse);
    }
}

void DFT::DFTExecutor::dftNd(float* output,
                             const VectorDims& outputShape,
                             const VectorDims& outputStrides,
                             std::vector<int32_t> axes,
                             bool inverse) const {
    const std::vector<size_t> iterationRange(outputShape.begin(), outputShape.end() - 1);
    const size_t lastDimIndex = iterationRange.size() - 1;
    for (size_t axisIndex = 0; axisIndex < axes.size(); ++axisIndex) {
        const size_t currentAxis = axes[axisIndex];
        const size_t outputComplexLen = outputShape[currentAxis];
        const size_t outputLen = outputComplexLen * 2;

        std::vector<size_t> iterationCounter(iterationRange.size(), 0);
        if (IsPowerOfTwo(outputComplexLen)) {
            size_t parallelDimIndex = lastDimIndex == currentAxis ? lastDimIndex - 1 : lastDimIndex;
            do {
                parallel_for(iterationRange[parallelDimIndex], [&](size_t dim) {
                    std::vector<float> gatheredData(outputLen);
                    auto parallelIterationCounter = iterationCounter;
                    parallelIterationCounter[parallelDimIndex] = dim;

                    gatherToBufferND(gatheredData.data(),
                                     output,
                                     currentAxis,
                                     parallelIterationCounter,
                                     outputShape,
                                     outputStrides);
                    fft(gatheredData.data(), outputLen, inverse);
                    applyBufferND(gatheredData.data(),
                                  output,
                                  currentAxis,
                                  parallelIterationCounter,
                                  outputShape,
                                  outputStrides);
                });
                iterationCounter[parallelDimIndex] = iterationRange[parallelDimIndex] - 1;
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        } else {
            std::vector<float> gatheredData(outputLen);
            do {
                gatherToBufferND(gatheredData.data(),
                                 output,
                                 currentAxis,
                                 iterationCounter,
                                 outputShape,
                                 outputStrides);
                naiveDFT(gatheredData.data(), outputLen, inverse);
                applyBufferND(gatheredData.data(), output, currentAxis, iterationCounter, outputShape, outputStrides);
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        }
    }
}

/* Cooley Tukey implementation of FFT */
void DFT::DFTRefExecutor::fft(float* data, int64_t dataLength, bool inverse, bool parallelize) const {
    static int cacheSizeL3 = dnnl::utils::get_cache_size(3, false);
    static int elementsPerCacheLine = cacheSizeL3 / sizeof(float);
    std::vector<float> bufferVector(dataLength * 2, 0);
    float* buffer = bufferVector.data();
    cpu_memcpy(buffer, data, dataLength * sizeof(float));

    size_t nComplex = dataLength / 2;
    float* inBufferStart = buffer + dataLength;
    float* outBufferStart = buffer;

    auto blockIteration = [&](const size_t numBlocks,
                               const size_t block,
                               const size_t blockSize,
                               const size_t nextIterationBlockSize) {
        float* curInpBufferPtr = inBufferStart + block * blockSize;
        float* curOutBufferPtr = outBufferStart + block * nextIterationBlockSize;
        float twiddleReal = twiddlesFFT[(numBlocks + block - 1) * 2];
        float twiddleImag = twiddlesFFT[(numBlocks + block) * 2 - 1];

        if (inverse) {
            twiddleImag *= -1;
        }

        for (size_t pair = 0; pair < blockSize / 2; pair += 2) {
            const float evenReal = curInpBufferPtr[pair];
            const float evenImag = curInpBufferPtr[pair + 1];

            const float oddReal = curInpBufferPtr[(blockSize / 2 + pair)];
            const float oddImag = curInpBufferPtr[(blockSize / 2 + pair) + 1];

            const float twiddledOddReal = getRealFromComplexProd(twiddleReal, twiddleImag, oddReal, oddImag);
            const float twiddledOddImag = getImaginaryFromComplexProd(twiddleReal, twiddleImag, oddReal, oddImag);

            curOutBufferPtr[pair] = evenReal + twiddledOddReal;
            curOutBufferPtr[pair + 1] = evenImag + twiddledOddImag;

            curOutBufferPtr[nComplex + pair] = evenReal - twiddledOddReal;
            curOutBufferPtr[nComplex + pair + 1] = evenImag - twiddledOddImag;
        }
    };

    size_t blockSize;
    size_t nextIterationBlockSize = dataLength;
    for (size_t numBlocks = 1; numBlocks < nComplex; numBlocks *= 2) {
        std::swap(inBufferStart, outBufferStart);

        blockSize = nextIterationBlockSize;
        nextIterationBlockSize /= 2;
        if (parallelize && blockSize >= 4 * elementsPerCacheLine) {
            parallel_for(numBlocks, [&](const size_t block) {
                blockIteration(numBlocks, block, blockSize, nextIterationBlockSize);
            });
        } else {
            for (size_t block = 0; block < numBlocks; ++block) {
                blockIteration(numBlocks, block, blockSize, nextIterationBlockSize);
            }
        }
    }

    for (int64_t k = 0; k < dataLength; k++) {
        if (inverse) {
            outBufferStart[k] /= nComplex;
        }
        data[k] = outBufferStart[k];
    }
}

void DFT::DFTRefExecutor::naiveDFT(float* data, size_t dataLength, bool inverse) const {
    std::vector<float> outputBuffer(dataLength);
    const size_t nComplex = dataLength / 2;
    const auto& twiddles = twiddlesMapDFT.find(nComplex)->second;

    parallel_for(nComplex, [&](size_t k) {
        float sumReal = 0.0f;
        float sumImag = 0.0f;
        for (size_t n = 0; n < nComplex; ++n) {
            auto complexRef = &twiddles[2 * (k * nComplex + n)];
            float complexReal = *complexRef;
            float complexImag = *(complexRef + 1);

            if (inverse) {
                complexImag *= -1;  // conjugate
            }
            float complexProdReal = getRealFromComplexProd(data[2 * n], data[2 * n + 1], complexReal, complexImag);
            float complexProdImag = getImaginaryFromComplexProd(data[2 * n], data[2 * n + 1], complexReal, complexImag);

            sumReal += complexProdReal;
            sumImag += complexProdImag;
        }

        if (inverse) {
            sumReal /= nComplex;
            sumImag /= nComplex;
        }
        outputBuffer[k * 2] = sumReal;
        outputBuffer[k * 2 + 1] = sumImag;
    });
    cpu_memcpy(data, outputBuffer.data(), dataLength * sizeof(float));
}

std::vector<float> DFT::DFTExecutor::generateTwiddlesDFT(size_t n_complex) const {
    std::vector<float> twiddles(n_complex * n_complex * 2);
    parallel_for(n_complex, [&](const size_t k) {
        for (size_t n = 0; n < n_complex; ++n) {
            float phase = 2.0f * PI * static_cast<float>(n * k) / static_cast<float>(n_complex);
            auto complexReal = std::cos(phase);
            auto complexImag = -std::sin(phase);
            twiddles[2 * (k * n_complex + n)] = complexReal;
            twiddles[2 * (k * n_complex + n) + 1] = complexImag;
        }
    });
    return twiddles;
}

void DFT::DFTExecutor::generateTwiddlesFFT(size_t n_complex) {
    size_t numBlocks = 1;

    twiddlesFFT.reserve((n_complex - 1) * 2);
    if (twiddlesFFT.size() == 0) {
        twiddlesFFT.emplace_back(1.0f);   //  cos(0)
        twiddlesFFT.emplace_back(-0.0f);  // -sin(0)
    } else {
        for (size_t i = numBlocks; i < twiddlesFFT.size() / 2; i += numBlocks) {
            numBlocks *= 2;
        }
    }

    for (size_t i = twiddlesFFT.size() / 2; i < n_complex - 1; i += numBlocks) {
        numBlocks *= 2;

        for (size_t blockNum = 0; blockNum < numBlocks; blockNum++) {
            size_t copyIndex = twiddlesFFT.size() - blockNum - numBlocks;

            twiddlesFFT.push_back(twiddlesFFT[copyIndex]);
            twiddlesFFT.push_back(twiddlesFFT[copyIndex + 1]);

            blockNum++;

            float angle = PI * blockNum / numBlocks;
            auto complexReal = std::cos(angle);
            auto complexImag = -std::sin(angle);

            twiddlesFFT.emplace_back(complexReal);
            twiddlesFFT.emplace_back(complexImag);
        }
    }
}

DFT::DFTJitExecutor::DFTJitExecutor(const DFTAttrs& interpAttrs) : DFTExecutor(interpAttrs) {
    jit_dft_config_params jdp = {};

    jdp.inverse = interpAttrs.inverse;

    if (mayiuse(cpu::x64::avx512_core)) {
        dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::avx512_core>(jdp));
        fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::avx512_core>(jdp));
    } else if (mayiuse(cpu::x64::avx2)) {
        dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::avx2>(jdp));
        fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::avx2>(jdp));
    } else if (mayiuse(cpu::x64::sse41)) {
        dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::sse41>(jdp));
        fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::sse41>(jdp));
    } else {
        IE_THROW() << "Can't create jit DFT kernel";
    }

    if (dftKernel)
        dftKernel->create_ker();
    if (fftKernel)
        fftKernel->create_ker();
}

void DFT::DFTJitExecutor::fft(float* data, int64_t dataLength, bool inverse, bool parallelize) const {
    static int cacheSizeL3 = dnnl::utils::get_cache_size(3, false);
    static int elementsPerCacheLine = cacheSizeL3 / sizeof(float);
    std::vector<float> bufferVector(dataLength * 2, 0);
    float* buffer = bufferVector.data();
    cpu_memcpy(buffer, data, dataLength * sizeof(float));

    size_t nComplex = dataLength / 2;
    float* inBufferStart = buffer + dataLength;
    float* outBufferStart = buffer;

    auto blockIteration = [&](const size_t block,
                              const size_t numBlocks,
                              const size_t blockSize,
                              const size_t nextIterationBlockSize) {
        auto arg = jit_args_fft();

        arg.src = inBufferStart + block * nextIterationBlockSize * 2;
        arg.dst = outBufferStart + block * nextIterationBlockSize;
        arg.twiddles = &twiddlesFFT[(numBlocks + block - 1) * 2];
        arg.num_blocks = numBlocks;
        arg.work_amount = nextIterationBlockSize;
        arg.n_complex = nComplex;

        (*fftKernel)(&arg);
    };

    size_t blockSize;
    size_t nextIterationBlockSize = dataLength;
    for (size_t numBlocks = 1; numBlocks < nComplex; numBlocks *= 2) {
        std::swap(inBufferStart, outBufferStart);

        blockSize = nextIterationBlockSize;
        nextIterationBlockSize /= 2;
        if (parallelize && blockSize >= 4 * elementsPerCacheLine) {
            parallel_for(numBlocks, [&](const size_t block) {
                blockIteration(block, 1, blockSize, nextIterationBlockSize);
            });
        } else {
            blockIteration(0, numBlocks, blockSize, nextIterationBlockSize);
        }
    }

    for (int64_t k = 0; k < dataLength; k++) {
        if (inverse) {
            outBufferStart[k] /= nComplex;
        }
        data[k] = outBufferStart[k];
    }
}

void DFT::DFTJitExecutor::naiveDFT(float* data, size_t dataLength, bool inverse) const {
    std::vector<float> outputBuffer(dataLength);
    const size_t nComplex = dataLength / 2;
    const auto& twiddles = twiddlesMapDFT.find(nComplex)->second;

    parallel_for(nComplex, [&](size_t k) {
        auto arg = jit_args_dft();

        arg.src = data;
        arg.dst = outputBuffer.data() + 2 * k;
        arg.twiddles = twiddles.data() + 2 * k * nComplex;
        arg.work_amount = nComplex;
        arg.index = k;

        (*dftKernel)(&arg);
    });
    cpu_memcpy(data, outputBuffer.data(), dataLength * sizeof(float));
}

bool DFT::created() const {
    return getType() == Type::DFT;
}

bool DFT::needPrepareParams() const {
    return inputShapesModified() || true;
}

void DFT::prepareParams() {
    DFTKey key = {};

    key.nodeAttrs.inverse = inverse;

    auto buildExecutor = [&](const DFTKey& key) -> std::shared_ptr<DFTExecutor> {
        std::shared_ptr<DFTExecutor> executor;
        if (mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<DFTJitExecutor>(key.nodeAttrs);
        } else {
            executor = std::make_shared<DFTRefExecutor>(key.nodeAttrs);
        }
        return executor;
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;
}

void DFT::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
