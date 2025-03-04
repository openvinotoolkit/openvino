// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>

#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset3.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {
#if defined(OPENVINO_ARCH_X86_64)
#    define GET_OFF(field) offsetof(jit_extract_image_patches_args, field)

template <cpu_isa_t isa>
struct jit_extract_image_patches_kernel : public jit_uni_extract_image_patches_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_extract_image_patches_kernel)

    explicit jit_extract_image_patches_kernel(jit_extract_image_patches_params jpp)
        : jit_uni_extract_image_patches_kernel(jpp),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_num_pads, ptr[reg_params + GET_OFF(h_lo_pad)]);
        mov(reg_h_hi_pad, ptr[reg_params + GET_OFF(h_hi_pad)]);
        mov(reg_w_lo_pad, ptr[reg_params + GET_OFF(w_lo_pad)]);
        mov(reg_w_hi_pad, ptr[reg_params + GET_OFF(w_hi_pad)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        mov(reg_src_incr, jpp.SH * jpp.IW * jpp.dtype_size);
        mov(reg_aux64, reg_w_hi_pad);
        mul_by_const(reg_aux64, reg_aux64_2, jpp.SW * jpp.dtype_size);
        sub(reg_src_incr, reg_aux64);

        mov(reg_aux64, reg_w_lo_pad);
        mul_by_const(reg_aux64, reg_aux64_2, jpp.SW * jpp.dtype_size);
        add(reg_src_incr, reg_aux64);
        add(reg_src, reg_aux64);

        mov(reg_ow_work_amount, reg_w_hi_pad);
        sub(reg_ow_work_amount, reg_w_lo_pad);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        if (mayiuse_gather) {
            mov(reg_aux64, gather_index_table);
            uni_vmovups(vmm_gather_index, ptr[reg_aux64]);
        }
        loop();

        this->postamble();

        if (mayiuse_gather) {
            prepare_table();
        }
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    bool mayiuse_gather = (mayiuse(x64::avx2) || mayiuse(x64::avx512_core)) && (jpp.dtype_size == 4);
    uint32_t vlen = cpu_isa_traits<isa>::vlen;
    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;
    reg64_t reg_oh_count = r10;
    reg64_t reg_ow_count = r11;
    reg64_t reg_num_pads = r12;
    reg64_t reg_src_incr = r13;
    reg64_t reg_aux64 = rax;
    reg64_t reg_w_hi_pad = r14;
    reg64_t reg_w_lo_pad = r15;
    reg64_t reg_h_hi_pad = rbp;
    reg64_t reg_aux64_2 = rbx;
    reg64_t reg_ow_work_amount = rsi;
    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xmm xmm = Xmm(0);
    Vmm vmm_zero = Vmm(1);  // reserved for pad
    Xbyak::Xmm xmm_aux = Xbyak::Xmm(2);
    Vmm vmm_gather_index = Vmm(3);
    Vmm vmm_gather_mask = Vmm(4);
    Opmask k_mask = Xbyak::Opmask(1);
    Xbyak::Label gather_index_table;

    inline void load_scalar(Vmm vmm_arg, const Xbyak::Address& op) {
        auto xmm_src = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
        case 4:
            uni_vmovss(vmm_arg, op);
            break;
        case 2:
            uni_vpinsrw(xmm_src, xmm_src, op, 0x0);
            break;
        case 1:
            uni_vpinsrb(xmm_src, xmm_src, op, 0x0);
            break;
        default:
            OPENVINO_THROW("The data type of size '", jpp.dtype_size, "' is not supported.");
        }
    }
    inline void store_scalar(const Xbyak::Address& op, Vmm vmm_arg) {
        auto xmm_dst = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
        case 4:
            uni_vmovss(op, vmm_arg);
            break;
        case 2:
            uni_vpextrw(op, xmm_dst, 0x0);
            break;
        case 1:
            uni_vpextrb(op, xmm_dst, 0x0);
            break;
        default:
            OPENVINO_THROW("The data type of size '", jpp.dtype_size, "' is not supported.");
        }
    }

    inline void pad_with_zeros(reg64_t& reg_num_pads_arg, reg64_t& reg_dst_arg) {
        Xbyak::Label main, tail, exit;
        L(main);
        {
            cmp(reg_num_pads_arg, jpp.block_size);
            jl(tail);
            uni_vmovups(ptr[reg_dst_arg], vmm_zero);
            add(reg_dst_arg, jpp.dtype_size * jpp.block_size);
            sub(reg_num_pads_arg, jpp.block_size);
            jmp(main);
        }
        L(tail);
        {
            cmp(reg_num_pads_arg, 0);
            jle(exit);
            store_scalar(ptr[reg_dst_arg], vmm_zero);
            add(reg_dst_arg, jpp.dtype_size);
            dec(reg_num_pads_arg);
            jmp(tail);
        }
        L(exit);
    }

    inline void custom_uni_vgatherdps(const Vmm& vmm_arg, reg64_t& mem_base, const Vmm& mem_offset, Vmm& vmm_mask) {
        switch (isa) {
        case x64::avx2:
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_arg, ptr[mem_base + mem_offset], vmm_mask);
            break;
        case x64::avx512_core:
            kxnord(k_mask, k_mask, k_mask);
            vgatherdps(vmm_arg | k_mask, ptr[mem_base + mem_offset]);
            break;
        case x64::sse41:
            emulate_gather(vmm_arg, mem_base);
            break;
        default:
            OPENVINO_THROW("Got unsupported instruction set.");
        }
    }

    inline void gather_src2vmm(const Vmm& vmm_arg, reg64_t& mem_base) {
        switch (jpp.dtype_size) {
        case 4:
            custom_uni_vgatherdps(vmm, mem_base, vmm_gather_index, vmm_gather_mask);
            break;
        case 2:
        case 1:
            emulate_gather(vmm_arg, mem_base);
            break;
        default:
            OPENVINO_THROW("The data type of size '", jpp.dtype_size, "' is not supported.");
        }
    }

    inline void emulate_gather(const Xbyak::Xmm& xmm_arg, reg64_t& mem_base, int xmm_offset = 0) {
        const int xmm_size = 16;  // bytes
        const int xmm_block_size = xmm_size / jpp.dtype_size;
        const int offset = xmm_offset * jpp.SW * jpp.dtype_size * xmm_block_size;
        for (int i = 0; i < xmm_block_size; i++) {
            Xbyak::Address addr = ptr[mem_base + i * jpp.SW * jpp.dtype_size + offset];
            switch (jpp.dtype_size) {
            case 4:
                uni_vpinsrd(xmm_arg, xmm_arg, addr, i);
                break;
            case 2:
                uni_vpinsrw(xmm_arg, xmm_arg, addr, i);
                break;
            case 1:
                uni_vpinsrb(xmm_arg, xmm_arg, addr, i);
                break;
            default:
                OPENVINO_THROW("The data type of size '", jpp.dtype_size, "' is not supported.");
            }
        }
    }
    inline void emulate_gather(const Xbyak::Ymm& ymm_arg, reg64_t& mem_base) {
        auto low_xmm = Xbyak::Xmm(ymm_arg.getIdx());
        emulate_gather(low_xmm, mem_base, 0);
        emulate_gather(xmm_aux, mem_base, 1);
        vinserti128(ymm_arg, ymm_arg, xmm_aux, 1);
    }

    inline void emulate_gather(const Xbyak::Zmm& zmm_arg, reg64_t& mem_base) {
        auto low_xmm = Xbyak::Xmm(zmm_arg.getIdx());
        emulate_gather(low_xmm, mem_base, 0);
        for (int i = 1; i < 4; i++) {
            emulate_gather(xmm_aux, mem_base, i);
            vinserti64x2(zmm_arg, zmm_arg, xmm_aux, i);
        }
    }

    void loop() {
        mov(reg_oh_count, reg_h_hi_pad);
        // reg_num_pads contains h_lo_pad at this point
        sub(reg_oh_count, reg_num_pads);

        Xbyak::Label ih_loop, ih_tail, ih_exit;
        Xbyak::Label iw_loop, iw_tail, iw_exit;
        if (jpp.need_padding) {
            mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
            pad_with_zeros(reg_num_pads, reg_dst);
        }
        L(ih_loop);
        {
            cmp(reg_oh_count, 0);
            jle(ih_exit, T_NEAR);
            if (jpp.need_padding) {
                mov(reg_num_pads, reg_w_lo_pad);
                pad_with_zeros(reg_num_pads, reg_dst);
            }
            mov(reg_ow_count, reg_ow_work_amount);
            L(iw_loop);
            {
                cmp(reg_ow_count, jpp.block_size);
                jle(iw_tail, T_NEAR);
                gather_src2vmm(vmm, reg_src);
                add(reg_src, jpp.SW * jpp.dtype_size * jpp.block_size);
                uni_vmovups(ptr[reg_dst], vmm);
                add(reg_dst, jpp.dtype_size * jpp.block_size);
                sub(reg_ow_count, jpp.block_size);
                jmp(iw_loop);
            }
            L(iw_tail);
            {
                cmp(reg_ow_count, 0);
                jle(iw_exit, T_NEAR);
                load_scalar(vmm, ptr[reg_src]);
                store_scalar(ptr[reg_dst], vmm);
                dec(reg_ow_count);
                add(reg_src, jpp.SW * jpp.dtype_size);
                add(reg_dst, jpp.dtype_size);
                jmp(iw_tail);
            }
            L(iw_exit);
            if (jpp.need_padding) {
                mov(reg_num_pads, jpp.OW);
                sub(reg_num_pads, reg_w_hi_pad);
                pad_with_zeros(reg_num_pads, reg_dst);
            }
            dec(reg_oh_count);
            add(reg_src, reg_src_incr);
            jmp(ih_loop, T_NEAR);
        }
        L(ih_exit);
        if (jpp.need_padding) {
            mov(reg_num_pads, jpp.OH);
            sub(reg_num_pads, reg_h_hi_pad);
            mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
            pad_with_zeros(reg_num_pads, reg_dst);
        }
    }

    void prepare_table() {
        align(64);
        L(gather_index_table);
        for (size_t i = 0; i < vlen / sizeof(int32_t); i++) {
            dd(i * jpp.SW * jpp.dtype_size);
        }
    }
};
#endif  // OPENVINO_ARCH_X86_64

bool ExtractImagePatches::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                               std::string& errorMessage) noexcept {
    try {
        auto extImgPatcher = ov::as_type_ptr<const ov::opset3::ExtractImagePatches>(op);
        if (!extImgPatcher) {
            errorMessage = "Only opset3 ExtractImagePatches operation is supported";
            return false;
        }
        const auto padValue = extImgPatcher->get_auto_pad();
        if (!one_of(padValue, ov::op::PadType::VALID, ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER)) {
            errorMessage = "Does not support pad type: " + ov::as_string(padValue);
            return false;
        }
        if (!everyone_is(2u,
                         extImgPatcher->get_sizes().size(),
                         extImgPatcher->get_strides().size(),
                         extImgPatcher->get_rates().size())) {
            errorMessage = "Doesn't support 'sizes', 'strides', 'rates', attributes with rank != 2";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
struct ExtractImagePatchesKey {
    VectorDims inDims;
    VectorDims outDims;
    VectorDims kSizes;
    VectorDims strides;
    VectorDims rates;
    ExtractImagePatches::ExtImgPatcherPadType padType;
    size_t prcSize;
    [[nodiscard]] size_t hash() const;
    bool operator==(const ExtractImagePatchesKey& rhs) const;
};

size_t ExtractImagePatchesKey::hash() const {
    using namespace dnnl::impl::primitive_hashing;
    using namespace dnnl::impl;
    size_t seed = 0;
    seed = get_vector_hash(seed, inDims);
    seed = get_vector_hash(seed, outDims);
    seed = get_vector_hash(seed, kSizes);
    seed = get_vector_hash(seed, strides);
    seed = get_vector_hash(seed, rates);
    seed = hash_combine(seed, padType);
    seed = hash_combine(seed, prcSize);
    return seed;
}

bool ExtractImagePatchesKey::operator==(const ExtractImagePatchesKey& rhs) const {
    bool result = inDims == rhs.inDims && outDims == rhs.outDims && kSizes == rhs.kSizes && strides == rhs.strides &&
                  rates == rhs.rates && padType == rhs.padType && prcSize == rhs.prcSize;
    return result;
}
}  // namespace

ExtractImagePatches::ExtractImagePatches(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto extImgPatcher = ov::as_type_ptr<const ov::opset3::ExtractImagePatches>(op);

    if (inputShapes.size() != 1 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input or output edges!",
                           " Input: ",
                           inputShapes.size(),
                           "); Output: ",
                           outputShapes.size());
    }

    if (getInputShapeAtPort(0).getRank() != 4) {
        THROW_CPU_NODE_ERR("must have 4D input tensor. Actual: ", getInputShapeAtPort(0).getRank());
    }

    if (getOutputShapeAtPort(0).getRank() != 4) {
        THROW_CPU_NODE_ERR("must have 4D output tensor. Actual: ", getOutputShapeAtPort(0).getRank());
    }

    if (extImgPatcher->get_auto_pad() == ov::op::PadType::VALID) {
        _auto_pad = ExtImgPatcherPadType::VALID;
    } else if (extImgPatcher->get_auto_pad() == ov::op::PadType::SAME_LOWER) {
        _auto_pad = ExtImgPatcherPadType::SAME_LOWER;
    } else if (extImgPatcher->get_auto_pad() == ov::op::PadType::SAME_UPPER) {
        _auto_pad = ExtImgPatcherPadType::SAME_UPPER;
    } else {
        THROW_CPU_NODE_ERR("has unsupported pad type: ", extImgPatcher->get_auto_pad());
    }

    _ksizes = extImgPatcher->get_sizes();
    ;
    _strides = extImgPatcher->get_strides();
    _rates = extImgPatcher->get_rates();
    if (_ksizes.size() != 2 || _strides.size() != 2 || _rates.size() != 2) {
        THROW_CPU_NODE_ERR("must have the following attributes with shape {2}: sizes, strides, rates.");
    }
}

void ExtractImagePatches::prepareParams() {
    const auto& srcMemPtr0 = getSrcMemoryAtPort(0);
    const auto& dstMemPtr = getDstMemoryAtPort(0);
    if (!srcMemPtr0 || !srcMemPtr0->isDefined()) {
        THROW_CPU_NODE_ERR("Input memory is undefined.");
    }
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Destination memory is undefined.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    const auto& in_dims = getParentEdgeAt(0)->getMemory().getStaticDims();
    const auto& out_dims = getChildEdgeAt(0)->getMemory().getStaticDims();
    const auto prcSize = getOriginalInputPrecisionAtPort(0).size();
    ExtractImagePatchesKey key = {in_dims, out_dims, _ksizes, _strides, _rates, _auto_pad, prcSize};
    const auto isJit = mayiuse(x64::sse41);
    auto buildExecutor = [&isJit](const ExtractImagePatchesKey& key) -> executorPtr {
        if (isJit) {
            return std::make_shared<ExtractImagePatchesJitExecutor>(key.inDims,
                                                                    key.outDims,
                                                                    key.kSizes,
                                                                    key.strides,
                                                                    key.rates,
                                                                    key.padType,
                                                                    key.prcSize);
        }
        return std::make_shared<ExtractImagePatchesRefExecutor>(key.inDims,
                                                                key.outDims,
                                                                key.kSizes,
                                                                key.strides,
                                                                key.rates,
                                                                key.padType,
                                                                key.prcSize);
    };
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;
}

void ExtractImagePatches::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const auto precision = getOriginalInputPrecisionAtPort(0);
    if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end()) {
        THROW_CPU_NODE_ERR("has unsupported precision: ", precision.get_type_name());
    }

    addSupportedPrimDesc({{LayoutType::ncsp, precision}}, {{LayoutType::ncsp, precision}}, impl_desc_type::ref_any);
}

void ExtractImagePatches::execute(const dnnl::stream& strm) {
    if (execPtr) {
        auto src = getSrcDataAtPort(0);
        auto dst = getDstDataAtPort(0);
        const auto inStrides = getParentEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto outStrides = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
        execPtr->exec(src, dst, inStrides, outStrides);
    } else {
        THROW_CPU_NODE_ERR("Primitive wasn't created");
    }
}

void ExtractImagePatches::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void ExtractImagePatches::ExtractImagePatchesRefExecutor::executeReference(void* src,
                                                                           void* dst,
                                                                           const VectorDims& istrides,
                                                                           const VectorDims& ostrides) const {
    const auto* src_data = reinterpret_cast<const char*>(src);
    auto* dst_data = reinterpret_cast<char*>(dst);

    const std::vector<size_t> ostrides_partial = {ostrides[0],
                                                  jpp.KW * IC * ostrides[1],
                                                  IC * ostrides[1],
                                                  ostrides[1]};

    parallel_for4d(OB, jpp.KH, jpp.KW, IC, [&](const size_t ob, const size_t kh, const size_t kw, const size_t ic) {
        const int64_t iw_start = static_cast<int64_t>(kw * RW) - PL;
        const int64_t ih_start = static_cast<int64_t>(kh * RH) - PT;
        const size_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(-1.f * ih_start / jpp.SH);
        const size_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(-1.f * iw_start / jpp.SW);

        const size_t ih_hpad =
            std::ceil((IH - 1.f * ih_start) / jpp.SH) > jpp.OH ? jpp.OH : std::ceil((IH + -1.f * ih_start) / jpp.SH);
        const size_t iw_hpad = std::ceil((jpp.IW - 1.f * iw_start) / jpp.SW) > jpp.OW
                                   ? jpp.OW
                                   : std::ceil((jpp.IW - 1.f * iw_start) / jpp.SW);

        char* my_dst_ptr = dst_data + (ob * ostrides_partial[0] + kh * ostrides_partial[1] + kw * ostrides_partial[2] +
                                       ic * ostrides_partial[3]) *
                                          jpp.dtype_size;
        const char* my_src_ptr =
            src_data + (ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start) * jpp.dtype_size;

        size_t num_bytes_to_set = ih_lpad * jpp.OW * jpp.dtype_size;
        memset(my_dst_ptr, 0, num_bytes_to_set);
        my_dst_ptr += num_bytes_to_set;

        const char* src_ptr_h_stop = my_src_ptr + ih_hpad * jpp.SH * jpp.IW * jpp.dtype_size;
        for (const char* src_h_ptr = my_src_ptr + ih_lpad * jpp.SH * jpp.IW * jpp.dtype_size;
             src_h_ptr < src_ptr_h_stop;
             src_h_ptr += jpp.SH * jpp.IW * jpp.dtype_size) {
            num_bytes_to_set = iw_lpad * jpp.dtype_size;
            memset(my_dst_ptr, 0, num_bytes_to_set);
            my_dst_ptr += num_bytes_to_set;

            const char* src_ptr_w_stop = src_h_ptr + iw_hpad * jpp.SW * jpp.dtype_size;
            for (const char* src_w_ptr = src_h_ptr + iw_lpad * jpp.SW * jpp.dtype_size; src_w_ptr < src_ptr_w_stop;
                 src_w_ptr += jpp.SW * jpp.dtype_size) {
                num_bytes_to_set = jpp.dtype_size;
                memcpy(my_dst_ptr, src_w_ptr, num_bytes_to_set);
                my_dst_ptr += num_bytes_to_set;
            }
            num_bytes_to_set = (jpp.OW - iw_hpad) * jpp.dtype_size;
            memset(my_dst_ptr, 0, num_bytes_to_set);
            my_dst_ptr += num_bytes_to_set;
        }
        num_bytes_to_set = (jpp.OH - ih_hpad) * jpp.OW * jpp.dtype_size;
        memset(my_dst_ptr, 0, num_bytes_to_set);
    });
}

void ExtractImagePatches::ExtractImagePatchesJitExecutor::executeOptimizedGeneric(void* src,
                                                                                  void* dst,
                                                                                  const VectorDims& istrides,
                                                                                  const VectorDims& ostrides) const {
#if defined(OPENVINO_ARCH_X86_64)
    const auto* src_data = reinterpret_cast<const char*>(src);
    auto* dst_data = reinterpret_cast<char*>(dst);
    const auto& jpp = pKernel->jpp;

    const std::vector<size_t> ostrides_partial = {ostrides[0],
                                                  jpp.KW * IC * ostrides[1],
                                                  IC * ostrides[1],
                                                  ostrides[1]};

    parallel_for4d(OB, jpp.KH, jpp.KW, IC, [&](const size_t ob, const size_t kh, const size_t kw, const size_t ic) {
        const int64_t ih_start = kh * RH - PT;
        const int64_t iw_start = kw * RW - PL;
        const size_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(-1.f * ih_start / jpp.SH);
        const size_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(-1.f * iw_start / jpp.SW);
        const size_t ih_hpad =
            std::ceil((IH - 1.f * ih_start) / jpp.SH) > jpp.OH ? jpp.OH : std::ceil((IH - 1.f * ih_start) / jpp.SH);
        const size_t iw_hpad = std::ceil((jpp.IW - 1.f * iw_start) / jpp.SW) > jpp.OW
                                   ? jpp.OW
                                   : std::ceil((jpp.IW - 1.f * iw_start) / jpp.SW);

        size_t dst_offset =
            ob * ostrides_partial[0] + kh * ostrides_partial[1] + kw * ostrides_partial[2] + ic * ostrides_partial[3];
        size_t src_offset =
            ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start + ih_lpad * jpp.SH * jpp.IW;

        auto args = jit_extract_image_patches_args();
        args.src = src_data + src_offset * jpp.dtype_size;
        args.dst = dst_data + dst_offset * jpp.dtype_size;
        args.h_lo_pad = ih_lpad;
        args.h_hi_pad = ih_hpad;
        args.w_lo_pad = iw_lpad;
        args.w_hi_pad = iw_hpad;
        (*pKernel)(&args);
    });
#endif  // OPENVINO_ARCH_X86_64
}

jit_extract_image_patches_params ExtractImagePatches::ExtractImagePatchesExecutor::fillJpp(
    const VectorDims& inDims,
    const VectorDims& outDims,
    const VectorDims& kSizes,
    const VectorDims& strides,
    const VectorDims& rates,
    const ExtImgPatcherPadType& padType,
    const size_t prcSize) {
    jit_extract_image_patches_params jpp{};

    IC = inDims[1];
    IH = inDims[2];
    jpp.IW = inDims[3];

    OB = outDims[0];
    jpp.OH = outDims[2];
    jpp.OW = outDims[3];

    jpp.KH = kSizes[0];
    jpp.KW = kSizes[1];

    jpp.SH = strides[0];
    jpp.SW = strides[1];

    RH = rates[0];
    RW = rates[1];

    PL = 0;
    PT = 0;
    jpp.need_padding = false;
    if (padType != ExtImgPatcherPadType::VALID) {
        const int64_t ihStep = kSizes[0] + (rates[0] - 1) * (kSizes[0] - 1);
        const int64_t iwStep = kSizes[1] + (rates[1] - 1) * (kSizes[1] - 1);

        int64_t PW = (std::ceil(1.f * jpp.IW / strides[1]) - 1) * strides[1] + iwStep - jpp.IW;
        int64_t PH = (std::ceil(1.f * IH / strides[0]) - 1) * strides[0] + ihStep - IH;

        int64_t increment_sign = 0;
        if (padType == ExtImgPatcherPadType::SAME_LOWER) {
            increment_sign = 1;
        } else if (padType == ExtImgPatcherPadType::SAME_UPPER) {
            increment_sign = -1;
        }

        if ((PW > 0) && (PW < iwStep)) {
            PL = static_cast<size_t>((PW + increment_sign * (PW % 2)) / 2);
            jpp.need_padding = true;
        }
        if ((PH > 0) && (PH < ihStep)) {
            PT = static_cast<size_t>((PH + increment_sign * (PH % 2)) / 2);
            jpp.need_padding = true;
        }
    }

    jpp.dtype_size = prcSize;
    if (mayiuse(x64::avx512_core)) {
        jpp.block_size = cpu_isa_traits<x64::avx512_core>::vlen / prcSize;
    } else if (mayiuse(x64::avx2)) {
        jpp.block_size = cpu_isa_traits<x64::avx2>::vlen / prcSize;
    } else if (mayiuse(x64::sse41)) {
        jpp.block_size = cpu_isa_traits<x64::sse41>::vlen / prcSize;
    } else {
        jpp.block_size = 1;
    }

    return jpp;
}

ExtractImagePatches::ExtractImagePatchesJitExecutor::ExtractImagePatchesJitExecutor(const VectorDims& inDims,
                                                                                    const VectorDims& outDims,
                                                                                    const VectorDims& kSizes,
                                                                                    const VectorDims& strides,
                                                                                    const VectorDims& rates,
                                                                                    const ExtImgPatcherPadType& padType,
                                                                                    const size_t prcSize) {
#if defined(OPENVINO_ARCH_X86_64)
    auto jpp = fillJpp(inDims, outDims, kSizes, strides, rates, padType, prcSize);
    if (mayiuse(x64::avx512_core)) {
        pKernel = std::make_unique<jit_extract_image_patches_kernel<x64::avx512_core>>(jpp);
    } else if (mayiuse(x64::avx2)) {
        pKernel = std::make_unique<jit_extract_image_patches_kernel<x64::avx2>>(jpp);
    } else if (mayiuse(x64::sse41)) {
        pKernel = std::make_unique<jit_extract_image_patches_kernel<x64::sse41>>(jpp);
    } else {
        OPENVINO_THROW("Can't create jit extract image patches kernel");
    }

    if (pKernel) {
        pKernel->create_ker();
    }
#endif  // OPENVINO_ARCH_X86_64
}

void ExtractImagePatches::ExtractImagePatchesJitExecutor::exec(void* src,
                                                               void* dst,
                                                               const VectorDims& istrides,
                                                               const VectorDims& ostrides) {
    if (!pKernel) {
        OPENVINO_THROW("Can't execute, kernel for extract image patches node is not compiled");
    }
    executeOptimizedGeneric(src, dst, istrides, ostrides);
}

ExtractImagePatches::ExtractImagePatchesRefExecutor::ExtractImagePatchesRefExecutor(const VectorDims& inDims,
                                                                                    const VectorDims& outDims,
                                                                                    const VectorDims& kSizes,
                                                                                    const VectorDims& strides,
                                                                                    const VectorDims& rates,
                                                                                    const ExtImgPatcherPadType& padType,
                                                                                    const size_t prcSize)
    : jpp(fillJpp(inDims, outDims, kSizes, strides, rates, padType, prcSize)) {}

void ExtractImagePatches::ExtractImagePatchesRefExecutor::exec(void* src,
                                                               void* dst,
                                                               const VectorDims& istrides,
                                                               const VectorDims& ostrides) {
    executeReference(src, dst, istrides, ostrides);
}

const std::set<size_t> ExtractImagePatches::_supported_precisions_sizes = {1, 2, 4};

bool ExtractImagePatches::created() const {
    return getType() == Type::ExtractImagePatches;
}

}  // namespace ov::intel_cpu::node
