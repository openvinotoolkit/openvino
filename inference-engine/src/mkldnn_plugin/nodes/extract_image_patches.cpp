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
    // The main loop where all the work is done
    void loop() {
        //Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;
        Xbyak::Label top_pad;
        //for (int64_t i = 0; i < ih_lpad * OW; i++)
        //    dst_data[dst_idx++] = T(0);

        //mov(reg_i, ptr[reg_params + GET_OFF(h_lo_pad)]);
        mov(reg_i, jpp.block_size);
        //mov(reg_i, ptr[reg_params + GET_OFF(h_hi_pad)]);
        uni_vmovups(vmm, 1);
        uni_vpxor(vmm, vmm, vmm);
        //pxor(vmm, vmm);
        L(top_pad);
        {
            cmp(reg_i, jpp.block_size);
            jl(exit_label, T_NEAR);
            uni_vmovups(ptr[reg_dst], vmm);
            add(reg_dst, jpp.dtype_size * jpp.block_size); // sure need to mult by dtype_size?
            sub(reg_i, jpp.block_size);
            jmp(top_pad, T_NEAR);
        }
        // set appropriate dst in case some extra zeros were written
        //mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        //mov(reg_i, ptr[reg_params + GET_OFF(h_lo_pad)]);
        //mul_by_const(reg_i, reg_aux, jpp.dtype_size * jpp.block_size);
        //add(reg_dst,  reg_i);
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
        /*
        mov(ih_work_amount, jpp.OH);
        L(main_ih_loop);
        {
            cmp(ih_work_amount, 1);
            jl(tail_loop_label, T_NEAR);

            mov(iw_work_amount, jpp.OW);
            L(main_iw_loop);
            {
                cmp(iw_work_amount, 1);
                jl(main_iw_loop_done, T_NEAR);

                //uni_vmovups(vmm, ptr[reg_src]);
                //uni_vmovups(ptr[reg_dst], vmm);

                add(reg_src, jpp.SW * jpp.data_size);
                add(reg_dst, jpp.data_size);

                sub(iw_work_amount, 1);
                jmp(main_iw_loop, T_NEAR);
            }
            L(main_iw_loop_done);

            sub(ih_work_amount, 1);
            jmp(main_ih_loop, T_NEAR);
        }
        */
        /*
        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            je(exit_label, T_NEAR);

            if (n + 1 == jpp.ndims) {
                load(xmm, ptr[reg_src]);
                store(ptr[reg_dst], xmm);
            } else {
                aux_reg_src = reg_src;
                aux_reg_dst = reg_dst;
                push(aux_reg_src);
                push(aux_reg_dst);
                push(reg_work_amount);
                loop(n + 1);
                pop(reg_work_amount);
                pop(reg_dst);
                pop(reg_src);
            }

            add(reg_src, jpp.src_strides[n] * jpp.data_size);
            add(reg_dst, jpp.dst_strides[n] * jpp.data_size);
            sub(reg_work_amount, 1);

            jmp(tail_loop_label, T_NEAR);
        }
        */
        L(exit_label);
    }


private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;
    reg64_t reg_work_amount = r10;
    reg64_t reg_i = r11;
    reg64_t reg_aux = r12;
    reg64_t reg_params = abi_param1;

    Vmm vmm = Vmm(0);
    Xbyak::Xmm xmm = Xbyak::Xmm(0);
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
    } catch (InferenceEngine::Exception &ex) {
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
