// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"
#include "utils/bfloat16.hpp"
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <nodes/eltwise.h>

using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& node, Precision exec_prc)
: jit_emitter(host, host_isa, node, exec_prc) {
    input_type = node->get_input_element_type(0);
    output_type = node->get_output_element_type(0);

    if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
       emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(host, host_isa));
}

size_t jit_convert_emitter::get_inputs_num() const { return 1; }

void jit_convert_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                                    const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                                    const emitter_context *emit_context) const {
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx512_common) {
        emit_isa<cpu::x64::avx512_common>(in_vec_idxs, out_vec_idxs);
    } else {
        assert(!"unsupported isa");
    }
}

void jit_convert_emitter::emit_data() const {
    if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core) && !!emu_vcvtneps2bf16)
        emu_vcvtneps2bf16->emit_data();
}

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst  = Vmm(out_vec_idxs[0]);

    if (input_type == output_type) {
        h->uni_vmovups(vmm_dst, vmm_src);
        return;
    }

    auto is_supported_type = [this](const ov::element::Type& type) {
        return any_of(supported_types.begin(), supported_types.end(),
            [&type](const ov::element::Type& supported_type) { return supported_type == type; } );
    };

    if (!is_supported_type(input_type))
        IE_THROW() << "Unsupported input type: " << input_type.get_type_name();
    if (!is_supported_type(output_type))
        IE_THROW() << "Unsupported output type: " << output_type.get_type_name();

    switch (input_type) {
        case ov::element::f32:
            break;
        case ov::element::bf16:
            h->vpmovzxwd(vmm_src, vmm_src);
            if (output_type == ov::element::f32) {
                h->uni_vpslld(vmm_dst, vmm_src, 16);
            } else {
                h->uni_vpslld(vmm_src, vmm_src, 16);
            }
            break;
        case ov::element::i8:
            h->uni_vpmovsxbd(vmm_src, vmm_src);
            break;
        case ov::element::u8:
            h->uni_vpmovzxbd(vmm_src, vmm_src);
            break;
        default:
            assert(!"unsupported output data type");
    }

    switch (output_type) {
        case ov::element::f32:
            if (input_type != ov::element::bf16) {
                h->uni_vcvtdq2ps(vmm_dst, vmm_src);
            }
            break;
        case ov::element::bf16:
            if (input_type != ov::element::f32) {
                h->uni_vcvtdq2ps(vmm_src, vmm_src);
            }
            float2bfloat({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            break;
        case ov::element::i8:
            if (input_type != ov::element::u8) {
                h->uni_vcvtps2dq(vmm_src, vmm_src);
            }
            dword2sint8<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            break;
        case ov::element::u8:
            if (input_type != ov::element::i8) {
                h->uni_vcvtps2dq(vmm_src, vmm_src);
            }
            dword2uint8<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            break;
        default:
            assert(!"unsupported output data type");
    }
}

void jit_convert_emitter::float2bfloat(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    Zmm zmm_src = Zmm(in_vec_idxs[0]);
    Zmm zmm_dst  = Zmm(out_vec_idxs[0]);

    if (mayiuse(avx512_core_bf16)) {
        h->vcvtneps2bf16(zmm_dst, zmm_src);
    } else {
        if (!emu_vcvtneps2bf16)
            IE_THROW() << "Converter from float to bf16 isn't initialized!";

        emu_vcvtneps2bf16->emit_code({static_cast<size_t>(zmm_src.getIdx())}, {static_cast<size_t>(zmm_dst.getIdx())});
    }
}

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_emitter::dword2sint8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Xmm xmm_src = Xmm(in_vec_idxs[0]);
    Ymm ymm_src = Ymm(in_vec_idxs[0]);

    Xmm xmm_dst = Xmm(out_vec_idxs[0]);

    if (isa == mkldnn::impl::cpu::x64::avx512_common) {
        h->vpmovsdb(xmm_dst, vmm_src);
    } else {
        h->uni_vpackssdw(vmm_src, vmm_src, vmm_src);
        if (isa != mkldnn::impl::cpu::x64::sse41)
            h->vpermq(ymm_src, ymm_src, 0x08);
        h->uni_vpacksswb(xmm_dst, xmm_src, xmm_src);
    }
}

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_emitter::dword2uint8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Xmm xmm_src = Xmm(in_vec_idxs[0]);
    Ymm ymm_src = Ymm(in_vec_idxs[0]);

    Xmm xmm_dst = Xmm(out_vec_idxs[0]);

    if (isa == mkldnn::impl::cpu::x64::avx512_common) {
        Vmm vmm_zero  = Vmm(out_vec_idxs[0]);
        h->vpxord(vmm_zero, vmm_zero, vmm_zero);
        h->vpmaxsd(vmm_src, vmm_src, vmm_zero);
        h->vpmovusdb(xmm_dst, vmm_src);
    } else {
        h->uni_vpackusdw(vmm_src, vmm_src, vmm_src);
        if (isa != mkldnn::impl::cpu::x64::sse41)
            h->vpermq(ymm_src, ymm_src, 0x08);
        h->uni_vpackuswb(xmm_dst, xmm_src, xmm_src);
    }
}

}   // namespace intel_cpu
}   // namespace ov
