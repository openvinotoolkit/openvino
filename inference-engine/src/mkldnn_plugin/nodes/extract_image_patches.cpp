// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches.hpp"
#include "list.hpp"
#include <cpu/x64/jit_generator.hpp>

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
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        loop();  //loop over input and output to get the work done

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;
    using reg8_t = const Xbyak::Reg8;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;

    reg64_t reg_i = r10;
    reg64_t reg_j = r11;
    reg64_t reg_num = r12; // reserved. used to specify padding
    reg64_t reg_aux64 = r13;
    reg32_t reg_aux32 = r13d;
    reg16_t reg_aux16 = r13w;
    reg8_t reg_aux8 = r13b;
    //reg64_t reg_aux64_2 = r14;
    reg32_t reg_aux32_2 = r14d;

    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xmm xmm = Xmm(0);
    Vmm vmm_zero = Vmm(1); // reserved for pad
    Xbyak::Xmm xmm_aux = Xbyak::Xmm(2);

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, Precision prec) {
        switch (prec) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(vmm_src, op);
                break;
            case Precision::U16:
                uni_vpmovzxwd(vmm_src, op);
                break;
            case Precision::I16:
                uni_vpmovsxwd(vmm_src, op);
                break;
            case Precision::I8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case Precision::U8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown precision");
        }
    }
    inline void load_scalar_size(Xmm xmm_src, const Xbyak::Address &op, Precision prec) {
        switch (jpp.dtype_size) {
            case 4: movss(xmm_src, op); break;
            case 2: pinsrw(xmm_src, op, 0); break;
            case 1: pinsrb(xmm_src, op, 0); break;
            default:
                assert(!"unknown precision");
        }
    }
    inline void load_scalar_sign(Xmm xmm_src, const Xbyak::Address &op, Precision prec) {
        switch (prec) {
            case Precision::FP32:
            case Precision::I32:
                movss(xmm_src, op);
                break;
            case Precision::I16:
                uni_vpmovsxwd(xmm_src, op); // NB! Why do we need to care about sign-extension manually& Isn't copy is just enough?
                break;
            case Precision::U16:
                uni_vpmovzxwd(xmm_src, op);
                break;
            case Precision::I8:
                movsx(reg_aux32, op);
                movq(xmm_src, reg_aux64); //NB! Why not movd or uni_vpinsrb
                break;
            case Precision::U8:
                movzx(reg_aux32, op);
                movq(xmm_src, reg_aux64);
                break;
            default:
                assert(!"unknown precision");
        }
    }
    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, Precision prec) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        switch (prec) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, vmm_dst);
                break;
            case Precision::I16:
                if (isa == x64::avx512_common) {
                    vpmovsdw(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::U16:
                if (isa == x64::avx512_common) {
                    vmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdw(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::I8:
                if (isa == x64::avx512_common) {
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case Precision::U8:
                if (isa == x64::avx512_common) {
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown precision");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, Precision prec) {
        //store_scalar_size(op, xmm_dst, prec);
        store_scalar_sign(op, xmm_dst, prec);
    }
    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, Precision prec) {
        //load_scalar_size(xmm_src, op, prec);
        load_scalar_sign(xmm_src, op, prec);
    }

    inline void store_scalar_size(const Xbyak::Address &op, Xmm xmm_dst, Precision prec) {
        switch (jpp.dtype_size) {
            case 4: movss(op, xmm_dst); break;
            case 2: pextrw(op, xmm_dst, 0); break;
            case 1: pextrb(op, xmm_dst, 0); break;
            default:
                assert(!"unknown dtype size");
        }
    }
    inline void store_scalar_sign(const Xbyak::Address &op, Xmm xmm_dst, Precision prec) {
        switch (prec) {
            case Precision::FP32:
            case Precision::I32:
                movss(op, xmm_dst);
                break;
            case Precision::I16:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_aux64, xmm_dst); // NB! Why copy the whole quadroword, but not just single word?
                mov(op, reg_aux8);
                break;
            case Precision::U16:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_aux64, xmm_dst);
                mov(op, reg_aux8);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_aux64, xmm_dst);
                mov(op, reg_aux8);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_aux64, xmm_dst);
                mov(op, reg_aux8);
                break;
            default:
                assert(!"unknown precision");
        }
    }
    // uses reg_num_pads as an argument
    // advances reg_dst
    void pad_with_zeros(reg64_t &reg_num_pads, reg64_t &reg_dst_arg) {
        Xbyak::Label main, tail, exit;
        L(main);
        {
            cmp(reg_num_pads, jpp.block_size);
            jl(tail);
            //uni_vmovups(ptr[reg_dst_arg], vmm_zero);
            store_vector(ptr[reg_dst_arg], vmm_zero, jpp.precision);
            add(reg_dst_arg, jpp.dtype_size * jpp.block_size);
            sub(reg_num_pads, jpp.block_size);
            jmp(main);
        }
        L(tail);
        {
            cmp(reg_num_pads, 0);
            jle(exit);
            //uni_vmovss(ptr[reg_dst_arg], vmm_zero);
            store_scalar(ptr[reg_dst_arg], vmm_zero, jpp.precision);
            add(reg_dst_arg, jpp.dtype_size);
            sub(reg_num_pads, 1);
            jmp(tail);
        }
        L(exit);
    }

    void read_src2xmm(const Xbyak::Xmm &xmm_arg, reg64_t &reg_src_arg) {
        const int xmm_size = 16; // bytes
        const int xmm_block_size = xmm_size / jpp.dtype_size;
        for (int i = 0; i < xmm_block_size; i++) {
            Xbyak::Address addr = ptr[reg_src_arg + i * jpp.SW * jpp.dtype_size];
            switch (jpp.dtype_size) {
                case 8: uni_vpinsrq(xmm_arg, xmm_arg, addr, i); break;
                case 4: uni_vpinsrd(xmm_arg, xmm_arg, addr, i); break;
                case 2: uni_vpinsrw(xmm_arg, xmm_arg, addr, i); break;
                case 1: uni_vpinsrb(xmm_arg, xmm_arg, addr, i); break;
            }
        }
        add(reg_src_arg, jpp.SW * jpp.dtype_size * xmm_block_size);
    }
    void read_src2vmm(Vmm vmm_arg, const Xbyak::Reg64 &reg_src_arg) {
        // isa == x64::sse41 || isa == x64::avx2 || isa == x64::avx512_common
        Xbyak::Xmm low_xmm = Xmm(vmm_arg.getIdx());
        read_src2xmm(low_xmm, reg_src_arg);
        if ((isa == x64::avx2) || (isa == x64::avx512_common)) {
            Xbyak::Ymm low_ymm = Xbyak::Ymm(vmm_arg.getIdx());
            read_src2xmm(xmm_aux, reg_src_arg);
            vinserti128(low_ymm, low_ymm, xmm_aux, 1);
            // NB! Need to implement further optimization for avx512?
            //if (isa == x64::avx512_common) {
            //}
        }
    }

    // The main loop where all the work is done
    void loop() {
        //for (int64_t i = 0; i < ih_lpad * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_num, ptr[reg_params + GET_OFF(h_lo_pad)]);
        mul_by_const(reg_num, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num, reg_dst);
        /*
        // The whole loop:
        for (int64_t ishift = ioffset + ih_lpad * SH * IW; ishift < ioffset + ih_hpad * SH * IW; ishift += SH * IW) {
            for (int64_t iw = 0; iw < iw_lpad; iw++)
                dst_data[dst_idx++] = 0;
            for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                dst_data[dst_idx++] = src_data[src_idx];

            for (int64_t i = 0; i < (OW - iw_hpad); i++)
                dst_data[dst_idx++] = 0;
        }
         */
        Xbyak::Label ih_loop, ih_tail, ih_exit;
        Xbyak::Label iw_loop, iw_tail, iw_exit;
        mov(reg_i, ptr[reg_params + GET_OFF(h_hi_pad)]);
        sub(reg_i, ptr[reg_params + GET_OFF(h_lo_pad)]);
        //for (int64_t ishift = ioffset + ih_lpad * SH * IW; ishift < ioffset + ih_hpad * SH * IW; ishift += SH * IW) {...}
        L(ih_loop);
        {
            //cmp(reg_i, jpp.block_size);
            cmp(reg_i, 0);
            jle(ih_exit, T_NEAR);
            /*
            for (int64_t iw = 0; iw < iw_lpad; iw++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_num, ptr[reg_params + GET_OFF(w_lo_pad)]);
            pad_with_zeros(reg_num, reg_dst);
            /*
            for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                dst_data[dst_idx++] = src_data[src_idx];
            */

            mov(reg_j, ptr[reg_params + GET_OFF(w_hi_pad)]);
            mov(reg_num, ptr[reg_params + GET_OFF(w_lo_pad)]);
            sub(reg_j, reg_num);
            mul_by_const(reg_num, reg_aux64, jpp.SW * jpp.dtype_size);
            add(reg_num, reg_src);
            L(iw_loop);
            {
                cmp(reg_j, jpp.block_size);
                jle(iw_tail, T_NEAR);
                read_src2vmm(vmm, reg_num);
                //uni_vmovups(ptr[reg_dst], vmm);
                store_vector(ptr[reg_dst], vmm, jpp.precision);
                add(reg_dst, jpp.dtype_size * jpp.block_size);
                sub(reg_j, jpp.block_size);
                jmp(iw_loop);
            }
            L(iw_tail);
            {
                cmp(reg_j, 0);
                jle(iw_exit, T_NEAR);
                //movss(xmm, ptr[reg_num]);
                load_scalar(xmm_aux, ptr[reg_num], jpp.precision);
                //movss(ptr[reg_dst], xmm);
                store_scalar(ptr[reg_dst], xmm_aux, jpp.precision);
                //mov(reg_aux32, ptr[reg_num]);
                //mov(ptr[reg_dst], reg_aux32);
                sub(reg_j, 1);
                add(reg_num, jpp.SW * jpp.dtype_size);
                add(reg_dst, jpp.dtype_size);
                jmp(iw_tail);
            }
            L(iw_exit);
            /*
            for (int64_t i = 0; i < (OW - iw_hpad); i++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_num, jpp.OW);
            sub(reg_num, ptr[reg_params + GET_OFF(w_hi_pad)]);
            pad_with_zeros(reg_num, reg_dst);
            sub(reg_i, 1);
            add(reg_src, jpp.SH * jpp.IW * jpp.dtype_size);
            jmp(ih_loop, T_NEAR);
        }
        L(ih_exit);
        //for (int64_t i = 0; i < (OH - ih_hpad) * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_num, jpp.OH);
        sub(reg_num, ptr[reg_params + GET_OFF(h_hi_pad)]);
        mul_by_const(reg_num, reg_aux64, jpp.OW);
        pad_with_zeros(reg_num, reg_dst);
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

        /*** JIT kernel configuration ***/
        jit_eximpat_params jcp;
        SizeVector in_dims = inData->getTensorDesc().getDims();
        jcp.OB = in_dims[0];
        jcp.IC = in_dims[1];
        jcp.IH = in_dims[2];
        jcp.IW = in_dims[3];
        SizeVector out_dims = layer->outData[0]->getTensorDesc().getDims();
        jcp.OH = out_dims[2];
        jcp.OW = out_dims[3];
        jcp.KH = _ksizes[0];
        jcp.KW = _ksizes[1];
        jcp.SH = _strides[0];
        jcp.SW = _strides[1];
        jcp.RH = _rates[0];
        jcp.RW = _rates[1];
        jcp.precision = layer->insData.front().lock()->getPrecision();
        jcp.dtype_size = jcp.precision.size();
        set_pads(_auto_pad, {jcp.IH, jcp.IW, jcp.KH, jcp.KW, jcp.SH, jcp.SW, jcp.RH, jcp.RW});
        jcp.PL = _pads[0];
        jcp.PT = _pads[1];
        jcp.block_size = 1;
        if (mayiuse(x64::avx512_common)) {
            jcp.block_size = cpu_isa_traits<x64::avx512_common>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx512_common>(jcp));
        } else if (mayiuse(x64::avx2)) {
            jcp.block_size = cpu_isa_traits<x64::avx2>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx2>(jcp));
        } else if (mayiuse(x64::sse41)) {
            jcp.block_size = cpu_isa_traits<x64::sse41>::vlen / jcp.dtype_size;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::sse41>(jcp));
        }

        if (eximpat_kernel)
            eximpat_kernel->create_ker();
        /*** JIT kernel configuration finished ***/

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
    switch (inputs[0]->getTensorDesc().getPrecision().size()) {
        case 1: {
            process_data<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs);
            break;
        }
        case 2: {
            process_data<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs);
            break;
        }
        case 4: {
            process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
            break;
        }
        case 8: {
            process_data<PrecisionTrait<Precision::U64>::value_type>(inputs, outputs);
            break;
        }
        default: {
            if (resp) {
                std::string errorMsg = "ExtractImagePatches layer does not support precision '"
                        + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }

    return OK;
}

void ExtractImagePatchesImpl::set_pads(const std::string & pad_str, const std::vector<int64_t> & params) {
    const int64_t IH = params[0]; const int64_t IW = params[1];
    const int64_t KH = params[2]; const int64_t KW = params[3];
    const int64_t SH = params[4]; const int64_t SW = params[5];
    const int64_t RH = params[6]; const int64_t RW = params[7];
    const int64_t iwStep = KW + (RW - 1) * (KW - 1);
    const int64_t ihStep = KH + (RH - 1) * (KH - 1);

    int64_t PL = 0, PT = 0;
    if (!CaselessEq<std::string>()(pad_str, "valid")) {
        int64_t PW = (std::ceil(1.f * IW/SW) - 1) * SW + iwStep - IW;
        int64_t PH = (std::ceil(1.f * IH/SH) - 1) * SH + ihStep - IH;

        if ((PW > 0) && (PW < iwStep)) {
            if (PW % 2 == 1) {
                if (CaselessEq<std::string>()(pad_str, "same_lower")) {
                    PL = (PW + 1) / 2;
                } else if (CaselessEq<std::string>()(pad_str, "same_upper")) {
                    PL = (PW - 1) / 2;
                }
            } else {
                PL = PW / 2;
            }
        }
        if ((PH > 0) && (PH < ihStep)) {
            if (PH % 2 == 1) {
                if (CaselessEq<std::string>()(pad_str, "same_lower")) {
                    PT = (PH + 1) / 2;
                } else if (CaselessEq<std::string>()(pad_str, "same_upper")) {
                    PT = (PH - 1) / 2;
                }
            } else {
                PT = PH / 2;
            }
        }
    }
    _pads.clear();
    _pads.push_back(PL);
    _pads.push_back(PT);
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
