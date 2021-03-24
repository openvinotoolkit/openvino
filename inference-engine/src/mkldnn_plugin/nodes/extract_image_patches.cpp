// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches.hpp"
#include "list.hpp"
#include <cpu/x64/jit_generator.hpp>
#include <cstring>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using details::CaselessEq;

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eximpat_args, field)

template <cpu_isa_t isa>
struct jit_uni_eximpat_kernel_f32 : public jit_uni_eximpat_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eximpat_kernel_f32)

    explicit jit_uni_eximpat_kernel_f32(jit_eximpat_params jpp) : jit_uni_eximpat_kernel(jpp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_w_hi_pad, ptr[reg_params + GET_OFF(w_hi_pad)]);
        mov(reg_w_lo_pad, ptr[reg_params + GET_OFF(w_lo_pad)]);
        mov(reg_h_hi_pad, ptr[reg_params + GET_OFF(h_hi_pad)]);

        mov(reg_src_h_incr, jpp.SH * jpp.IW * jpp.dtype_size);
        mov(reg_src_w_incr, reg_w_hi_pad);
        mul_by_const(reg_src_w_incr, reg_aux64, jpp.SW * jpp.dtype_size);
        sub(reg_src_h_incr, reg_src_w_incr);

        mov(reg_src_w_incr, reg_w_lo_pad);
        mul_by_const(reg_src_w_incr, reg_aux64, jpp.SW * jpp.dtype_size);

        mov(reg_ow_work_amount, reg_w_hi_pad);
        sub(reg_ow_work_amount, reg_w_lo_pad);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        loop();

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;
    reg64_t reg_oh_count = r10;
    reg64_t reg_ow_count = r11;
    reg64_t reg_num_pads = r12;
    reg64_t reg_src_h_incr = r13;
    reg64_t reg_aux64 = rax;
    reg64_t reg_w_hi_pad = r14;
    reg64_t reg_w_lo_pad = r15;
    reg64_t reg_h_hi_pad = rbp;
    reg64_t reg_src_w_incr = rbx;
    reg64_t reg_ow_work_amount = rsi;
    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xmm xmm = Xmm(0);
    Vmm vmm_zero = Vmm(1); // reserved for pad
    Xbyak::Xmm xmm_aux = Xbyak::Xmm(2);

    inline void load_scalar(Vmm vmm_arg, const Xbyak::Address &op) {
        Xbyak::Xmm xmm_src = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
            case 8: uni_vmovsd(xmm_src, op); break;
            case 4: uni_vmovss(xmm_src, op); break;
            case 2: uni_vpinsrw(xmm_src, xmm_src, op, 0x0); break;
            case 1: uni_vpinsrb(xmm_src, xmm_src, op, 0x0); break;
            default:
                assert(!"unknown dtype size");
        }
    }
    inline void store_scalar(const Xbyak::Address &op, Vmm vmm_arg) {
        Xbyak::Xmm xmm_dst = Xmm(vmm_arg.getIdx());
        switch (jpp.dtype_size) {
            case 8: uni_vmovsd(op, xmm_dst); break;
            case 4: uni_vmovss(op, xmm_dst); break;
            case 2: uni_vpextrw(op, xmm_dst, 0x0); break;
            case 1: uni_vpextrb(op, xmm_dst, 0x0); break;
            default:
                assert(!"unknown dtype size");
        }
    }

    inline void pad_with_zeros(reg64_t &reg_num_pads_arg, reg64_t &reg_dst_arg) {
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

    inline void read_src2vmm(const Xbyak::Xmm &xmm_arg, reg64_t &reg_src_arg) {
        const int xmm_size = 16; // bytes
        const int xmm_block_size = xmm_size / jpp.dtype_size;
        for (int i = 0; i < xmm_block_size; i++) {
            Xbyak::Address addr = ptr[reg_src_arg + i * jpp.SW * jpp.dtype_size];
            switch (jpp.dtype_size) {
                case 8: uni_vpinsrq(xmm_arg, xmm_arg, addr, i); break;
                case 4: uni_vpinsrd(xmm_arg, xmm_arg, addr, i); break;
                case 2: uni_vpinsrw(xmm_arg, xmm_arg, addr, i); break;
                case 1: uni_vpinsrb(xmm_arg, xmm_arg, addr, i); break;
                default:
                    assert(!"unknown dtype size");
            }
        }
        add(reg_src_arg, jpp.SW * jpp.dtype_size * xmm_block_size);
    }
    inline void read_src2vmm(const Xbyak::Ymm &ymm_arg, reg64_t &reg_src_arg) {
        Xbyak::Xmm low_xmm = Xmm(ymm_arg.getIdx());
        read_src2vmm(low_xmm, reg_src_arg);
        read_src2vmm(xmm_aux, reg_src_arg);
        vinserti128(ymm_arg, ymm_arg, xmm_aux, 1);
    }

    inline void read_src2vmm(const Xbyak::Zmm &zmm_arg, reg64_t &reg_src_arg) {
        Xbyak::Xmm low_xmm = Xmm(zmm_arg.getIdx());
        read_src2vmm(low_xmm, reg_src_arg);
        for (int i = 1; i < 4; i++) {
            read_src2vmm(xmm_aux, reg_src_arg);
            vinserti64x2(zmm_arg, zmm_arg, xmm_aux, i);
        }
    }

    void loop() {
        Xbyak::Label ih_loop, ih_tail, ih_exit;
        Xbyak::Label iw_loop, iw_tail, iw_exit;
        mov(reg_oh_count, reg_h_hi_pad);
        mov(reg_num_pads, ptr[reg_params + GET_OFF(h_lo_pad)]);
        sub(reg_oh_count, reg_num_pads);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num_pads, reg_dst);
        L(ih_loop);
        {
            cmp(reg_oh_count, 0);
            jle(ih_exit, T_NEAR);
            mov(reg_num_pads, reg_w_lo_pad);
            pad_with_zeros(reg_num_pads, reg_dst);
            mov(reg_ow_count, reg_ow_work_amount);
            add(reg_src, reg_src_w_incr);
            L(iw_loop);
            {
                cmp(reg_ow_count, jpp.block_size);
                jle(iw_tail, T_NEAR);
                read_src2vmm(vmm, reg_src);
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
            mov(reg_num_pads, jpp.OW);
            sub(reg_num_pads, reg_w_hi_pad);
            pad_with_zeros(reg_num_pads, reg_dst);
            dec(reg_oh_count);
            add(reg_src, reg_src_h_incr);
            jmp(ih_loop, T_NEAR);
        }
        L(ih_exit);
        mov(reg_num_pads, jpp.OH);
        sub(reg_num_pads, reg_h_hi_pad);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num_pads, reg_dst);
    }
};

ExtractImagePatchesImpl::ExtractImagePatchesImpl(const CNNLayer* layer) {
    try {
        std::string errorPrefix = std::string("Layer ") + layer->type + " with name '" + layer->name + "' ";
        if (details::CaselessEq<std::string>()("ExtractImagePatchesLayer", layer->type))
            IE_THROW() << errorPrefix << "is not an instance of ExtractImagePatchesLayer class";

        if (layer->insData.size() != 1 || layer->outData.size() != 1)
            IE_THROW() << errorPrefix << "has incorrect number of input or output edges!"
                << " Input: " << layer->insData.size() << "; Output: " << layer->outData.size();

        auto inData = layer->insData[0].lock();
        if (inData == nullptr)
            IE_THROW() << errorPrefix << "has nullable input data";

        if (inData->getTensorDesc().getDims().size() != 4)
            IE_THROW() << errorPrefix << "must have 4D input tensor. Actual: " << inData->getTensorDesc().getDims().size();

        if (layer->outData[0]->getTensorDesc().getDims().size() != 4)
            IE_THROW() << errorPrefix << "must have 4D output tensor. Actual: " << layer->outData[0]->getTensorDesc().getDims().size();

        if (inData->getLayout() != NCHW)
            IE_THROW() << errorPrefix << "has unsupported layout: " << inData->getLayout();

        const auto precision = inData->getTensorDesc().getPrecision();
        if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
            IE_THROW() << errorPrefix << "has unsupported precision: " << precision.name();

        auto ksizes = layer->GetParamAsUInts("sizes");
        auto strides = layer->GetParamAsUInts("strides");
        auto rates = layer->GetParamAsUInts("rates");
        _auto_pad = layer->GetParamAsString("auto_pad");
        if (!CaselessEq<std::string>()(_auto_pad, "valid")
                && !CaselessEq<std::string>()(_auto_pad, "same_upper")
                && !CaselessEq<std::string>()(_auto_pad, "same_lower"))
            IE_THROW() <<  errorPrefix << "has unsupported auto_pad value: " << _auto_pad;
        if (ksizes.size() != 2 || strides.size() != 2 || rates.size() != 2)
            IE_THROW() << errorPrefix << "must have the following attributes with shape {2}: sizes, strides, rates.";

        _ksizes.clear();
        _strides.clear();
        _rates.clear();
        for (size_t i = 0; i < ksizes.size(); i++)
            _ksizes.push_back((int64_t)ksizes[i]);
        for (size_t i = 0; i < strides.size(); i++)
            _strides.push_back((int64_t)strides[i]);
        for (size_t i = 0; i < rates.size(); i++)
            _rates.push_back((int64_t)rates[i]);
        jit_eximpat_params jpp;
        SizeVector in_dims = inData->getTensorDesc().getDims();
        jpp.IW = in_dims[3];
        SizeVector out_dims = layer->outData[0]->getTensorDesc().getDims();
        jpp.OH = out_dims[2];
        jpp.OW = out_dims[3];
        jpp.KH = _ksizes[0];
        jpp.KW = _ksizes[1];
        jpp.SH = _strides[0];
        jpp.SW = _strides[1];
        jpp.dtype_size = layer->insData.front().lock()->getPrecision().size();
        jpp.block_size = 1;

        if (mayiuse(x64::avx512_common)) {
            jpp.block_size = cpu_isa_traits<x64::avx512_common>::vlen / jpp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx512_common>(jpp));
        } else if (mayiuse(x64::avx2)) {
            jpp.block_size = cpu_isa_traits<x64::avx2>::vlen / jpp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx2>(jpp));
        } else if (mayiuse(x64::sse41)) {
            jpp.block_size = cpu_isa_traits<x64::sse41>::vlen / jpp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::sse41>(jpp));
        }

        if (eximpat_kernel)
            eximpat_kernel->create_ker();

        LayerConfig config;

        DataConfig inConfig;
        inConfig.desc = inData->getTensorDesc();
        config.inConfs.push_back(inConfig);

        DataConfig outConfig;
        outConfig.desc = layer->outData[0]->getTensorDesc();
        outConfig.desc.setPrecision(inConfig.desc.getPrecision());
        outConfig.desc.setLayout(inConfig.desc.getLayout());
        config.outConfs.push_back(outConfig);

        config.dynBatchSupport = false;
        confs.push_back(config);
    } catch (InferenceEngine::Exception &ex) {
        errorMsg = ex.what();
    }
}


StatusCode ExtractImagePatchesImpl::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
    const char *src_data = inputs[0]->cbuffer().as<const char *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    char *dst_data = outputs[0]->buffer().as<char *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    const int64_t dtype_size = inputs[0]->getTensorDesc().getPrecision().size();

    const auto& inDims = inputs[0]->getTensorDesc().getDims();
    const int64_t IC = inDims[1];
    const int64_t IH = inDims[2];
    const int64_t IW = inDims[3];

    const auto& outDims = outputs[0]->getTensorDesc().getDims();
    const int64_t OB = outDims[0];
    const int64_t OH = outDims[2];
    const int64_t OW = outDims[3];

    const int64_t KH = _ksizes[0], KW = _ksizes[1];
    const int64_t SH = _strides[0], SW = _strides[1];
    const int64_t RH = _rates[0], RW = _rates[1];

    int64_t pad_left = 0, pad_top = 0;
    if (!CaselessEq<std::string>()(_auto_pad, "valid")) {
        const int64_t iwStep = KW + (RW - 1) * (KW - 1);
        const int64_t ihStep = KH + (RH - 1) * (KH - 1);
        int64_t PW = (std::ceil(1.f * IW/SW) - 1) * SW + iwStep - IW;
        int64_t PH = (std::ceil(1.f * IH/SH) - 1) * SH + ihStep - IH;

        int64_t increment_sign = 0;
        if (CaselessEq<std::string>()(_auto_pad, "same_lower")) {
            increment_sign = 1;
        } else if (CaselessEq<std::string>()(_auto_pad, "same_upper")) {
            increment_sign = -1;
        }

        if ((PW > 0) && (PW < iwStep)) {
            pad_left = (PW + increment_sign * (PW % 2) ) / 2;
        }
        if ((PH > 0) && (PH < ihStep)) {
            pad_top = (PH + increment_sign * (PH % 2) ) / 2;
        }
    }
    const int64_t PL = pad_left;
    const int64_t PT = pad_top;

    const std::vector<int64_t> ostrides = {KH * KW * IC * OH * OW, KW * IC * OH * OW, IC * OH * OW, OH * OW};
    const std::vector<int64_t> istrides = {IC * IH * IW, IH * IW, IW};
    if (eximpat_kernel) {
        auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
            const int64_t ih_start = kh * RH - PT;
            const int64_t iw_start = kw * RW - PL;
            const int64_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(- 1.f * ih_start / SH);
            const int64_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(- 1.f * iw_start / SW);
            const int64_t ih_hpad = std::ceil((IH - 1.f * ih_start) / SH) > OH ? OH : std::ceil((IH - 1.f * ih_start) / SH);
            const int64_t iw_hpad = std::ceil((IW - 1.f * iw_start) / SW) > OW ? OW : std::ceil((IW - 1.f * iw_start) / SW);

            int64_t dst_offset = ob * ostrides[0] + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];
            int64_t src_offset = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start + ih_lpad * SH * IW;

            auto args = jit_eximpat_args();
            args.src = src_data + src_offset * dtype_size;
            args.dst = dst_data + dst_offset * dtype_size;
            args.h_lo_pad = ih_lpad;
            args.h_hi_pad = ih_hpad;
            args.w_lo_pad = iw_lpad;
            args.w_hi_pad = iw_hpad;
            (*eximpat_kernel)(&args);
        };
        parallel_for4d(OB, KH, KW, IC, thread_body);
    } else {
        auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
            const int64_t iw_start = kw * RW - PL;
            const int64_t ih_start = kh * RH - PT;
            const int64_t ih_lpad = ih_start >= 0 ? 0 : std::ceil(- 1.f * ih_start / SH);
            const int64_t iw_lpad = iw_start >= 0 ? 0 : std::ceil(- 1.f * iw_start / SW);

            const int64_t ih_hpad = std::ceil((IH - 1.f * ih_start) / SH) > OH ? OH : std::ceil((IH + -1.f * ih_start) / SH);
            const int64_t iw_hpad = std::ceil((IW - 1.f * iw_start) / SW) > OW ? OW : std::ceil((IW - 1.f * iw_start) / SW);

            char *my_dst_ptr = dst_data + (ob * ostrides[0] + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3]) * dtype_size;
            const char *my_src_ptr = src_data + (ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2] + iw_start) * dtype_size;

            int64_t num_bytes_to_set = ih_lpad * OW * dtype_size;
            memset(my_dst_ptr, 0, num_bytes_to_set);
            my_dst_ptr += num_bytes_to_set;

            const char* src_ptr_h_stop = my_src_ptr + ih_hpad * SH * IW * dtype_size;
            for (const char *src_h_ptr = my_src_ptr + ih_lpad * SH * IW * dtype_size;
                src_h_ptr < src_ptr_h_stop; src_h_ptr += SH * IW * dtype_size) {
                num_bytes_to_set = iw_lpad * dtype_size;
                memset(my_dst_ptr, 0, num_bytes_to_set);
                my_dst_ptr += num_bytes_to_set;

                const char* src_ptr_w_stop = src_h_ptr + iw_hpad * SW * dtype_size;
                for (const char* src_w_ptr = src_h_ptr + iw_lpad * SW * dtype_size;
                    src_w_ptr < src_ptr_w_stop; src_w_ptr += SW * dtype_size) {
                    num_bytes_to_set = dtype_size;
                    memcpy(my_dst_ptr, src_w_ptr, num_bytes_to_set);
                    my_dst_ptr += num_bytes_to_set;
                }
                num_bytes_to_set = (OW - iw_hpad) * dtype_size;
                memset(my_dst_ptr, 0, num_bytes_to_set);
                my_dst_ptr += num_bytes_to_set;
            }
            num_bytes_to_set = (OH - ih_hpad) * OW * dtype_size;
            memset(my_dst_ptr, 0, num_bytes_to_set);
        };
        parallel_for4d(OB, KH, KW, IC, thread_body);
    }
    return OK;
}
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
