// Copyright (C) 2021 Intel Corporation
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


        loop();  //loop over input and output to get the work done

        this->postamble();
    }
    // some additional functions to facilitate the main loop
    /*
    void load(const Xbyak::Xmm &xmm, const Xbyak::Address &addr) {
        switch (jpp.data_size) {
            case 16: movups(xmm, addr); break;
            case 8: movsd(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 2: pinsrw(xmm, addr, 0x0); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
        }
    }

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &xmm) {
        switch (jpp.data_size) {
            case 16: movups(addr, xmm); break;
            case 8: movsd(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 2: pextrw(addr, xmm, 0x0); break;
            case 1: pextrb(addr, xmm, 0x0); break;
        }
    }
    */
    // uses reg_num_pads as an argument
    // advances reg_dst
    void pad_with_zeros() {
        uni_vpxor(vmm, vmm, vmm);
        Xbyak::Label main, tail, exit;
        L(main);
        {
            cmp(reg_num_pads, jpp.block_size);
            jl(tail, T_NEAR);
            uni_vmovups(ptr[reg_dst], vmm);
            add(reg_dst, jpp.dtype_size * jpp.block_size);
            sub(reg_num_pads, jpp.block_size);
            jmp(main, T_NEAR);
        }
        L(tail);
        {
            cmp(reg_num_pads, 0);
            jle(exit, T_NEAR);
            movss(ptr[reg_dst], vmm); // will it work for 64 types?
            add(reg_dst, jpp.dtype_size);
            sub(reg_num_pads, 1);
            jmp(tail, T_NEAR);
        }
        L(exit);
    }
    // The main loop where all the work is done
    void loop() {
        //for (int64_t i = 0; i < ih_lpad * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_num_pads, ptr[reg_params + GET_OFF(h_lo_pad)]);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros();
        /*
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
        L(ih_loop);
        {
            //cmp(reg_i, jpp.block_size);
            cmp(reg_i, 0);
            jle(ih_exit, T_NEAR);
            /*
            for (int64_t iw = 0; iw < iw_lpad; iw++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_num_pads, ptr[reg_params + GET_OFF(w_lo_pad)]);
            pad_with_zeros();
            /*
            for (int64_t src_idx = ishift + iw_lpad * SW; src_idx < ishift + iw_hpad * SW; src_idx += SW)
                dst_data[dst_idx++] = src_data[src_idx];
            */

            mov(reg_j, ptr[reg_params + GET_OFF(w_lo_pad)]);
            mul_by_const(reg_j, reg_aux64, jpp.SW * jpp.dtype_size);
            add(reg_j, reg_src);

            mov(reg_num_pads, ptr[reg_params + GET_OFF(w_hi_pad)]);
            mul_by_const(reg_num_pads, reg_aux64, jpp.SW * jpp.dtype_size);
            add(reg_num_pads, reg_src);

            L(iw_loop);
            {
                cmp(reg_j, reg_num_pads);
                jge(iw_exit, T_NEAR);
                //uni_vmovups(vmm, ptr[reg_src]);
                //uni_vmovups(ptr[reg_dst], vmm);
                //add(reg_src, jpp.dtype_size * jpp.block_size);
                //add(reg_dst, jpp.dtype_size * jpp.block_size);
                mov(reg_aux32, ptr[reg_j]);
                mov(ptr[reg_dst], reg_aux32);
                //movss(vmm, ptr[reg_src]);
                //movss(ptr[reg_dst], vmm);

                add(reg_j, jpp.SW * jpp.dtype_size);
                add(reg_dst, jpp.dtype_size);
                jmp(iw_loop);
            }
            L(iw_exit);

            /*
            for (int64_t i = 0; i < (OW - iw_hpad); i++)
                dst_data[dst_idx++] = 0;
            */
            mov(reg_num_pads, jpp.OW);
            sub(reg_num_pads, ptr[reg_params + GET_OFF(w_hi_pad)]);
            pad_with_zeros();
            sub(reg_i, 1);
            add(reg_src, jpp.SH * jpp.IW * jpp.dtype_size);
            //sub(reg_i, jpp.block_size);
            jmp(ih_loop, T_NEAR);
        }
        L(ih_exit);
        //for (int64_t i = 0; i < (OH - ih_hpad) * OW; i++)
        //    dst_data[dst_idx++] = T(0);
        mov(reg_num_pads, jpp.OH);
        sub(reg_num_pads, ptr[reg_params + GET_OFF(h_hi_pad)]);
        mul_by_const(reg_num_pads, reg_aux64, jpp.OW);
        pad_with_zeros();
    }


private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src = r8;
    reg64_t reg_src_tmp = r10;
    reg64_t reg_dst = r9;

    reg64_t reg_i = r11;
    reg64_t reg_num_pads = r12; // reserved. used to specify padding
    reg64_t reg_aux64 = r13;
    reg64_t reg_j = r14;
    reg32_t reg_aux32 = r15d;
    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xbyak::Xmm xmm = Xbyak::Xmm(0);
};

ExtractImagePatchesImpl::ExtractImagePatchesImpl(const CNNLayer* layer) {
    try {
        std::string errorPrefix = std::string("Layer ") + layer->type + " with name '" + layer->name + "' ";
        if (details::CaselessEq<std::string>()("ExtractImagePatchesLayer", layer->type))
            THROW_IE_EXCEPTION << errorPrefix << "is not an instance of ExtractImagePatchesLayer class";

        if (layer->insData.size() != 1 || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of input or output edges!"
                << " Input: " << layer->insData.size() << "; Output: " << layer->outData.size();

        auto inData = layer->insData[0].lock();
        if (inData == nullptr)
            THROW_IE_EXCEPTION << errorPrefix << "has nullable input data";

        if (inData->getTensorDesc().getDims().size() != 4)
            THROW_IE_EXCEPTION << errorPrefix << "must have 4D input tensor. Actual: " << inData->getTensorDesc().getDims().size();

        if (layer->outData[0]->getTensorDesc().getDims().size() != 4)
            THROW_IE_EXCEPTION << errorPrefix << "must have 4D output tensor. Actual: " << layer->outData[0]->getTensorDesc().getDims().size();

        if (inData->getLayout() != NCHW)
            THROW_IE_EXCEPTION << errorPrefix << "has unsupported layout: " << inData->getLayout();

        const auto precision = inData->getTensorDesc().getPrecision();
        if (_supported_precisions_sizes.find(precision.size()) == _supported_precisions_sizes.end())
            THROW_IE_EXCEPTION << errorPrefix << "has unsupported precision: " << precision.name();

        auto ksizes = layer->GetParamAsUInts("sizes");
        auto strides = layer->GetParamAsUInts("strides");
        auto rates = layer->GetParamAsUInts("rates");
        _auto_pad = layer->GetParamAsString("auto_pad");
        if (!CaselessEq<std::string>()(_auto_pad, "valid")
                && !CaselessEq<std::string>()(_auto_pad, "same_upper")
                && !CaselessEq<std::string>()(_auto_pad, "same_lower"))
            THROW_IE_EXCEPTION <<  errorPrefix << "has unsupported auto_pad value: " << _auto_pad;
        if (ksizes.size() != 2 || strides.size() != 2 || rates.size() != 2)
            THROW_IE_EXCEPTION << errorPrefix << "must have the following attributes with shape {2}: sizes, strides, rates.";

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
        jcp.dtype_size = layer->insData.front().lock()->getPrecision().size();
        set_pads(_auto_pad, {jcp.IH, jcp.IW, jcp.KH, jcp.KW, jcp.SH, jcp.SW, jcp.RH, jcp.RW});
        jcp.PL = _pads[0];
        jcp.PT = _pads[1];
        jcp.block_size = 1;
        if (mayiuse(x64::avx512_common)) {
            jcp.block_size = 16;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx512_common>(jcp));
        } else if (mayiuse(x64::avx2)) {
            jcp.block_size = 8;
            eximpat_kernel.reset(new jit_uni_eximpat_kernel_f32<x64::avx2>(jcp));
        } else if (mayiuse(x64::sse41)) {
            jcp.block_size = 4;
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
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
    }
}

StatusCode ExtractImagePatchesImpl::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision().size()) {
        case 1: {
            process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
            break;
        }
        case 2: {
            process_data<PrecisionTrait<Precision::U16>::value_type>(inputs, outputs);
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
