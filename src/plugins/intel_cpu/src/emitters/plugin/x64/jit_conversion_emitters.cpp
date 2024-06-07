// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"

#include "utils/bfloat16.hpp"


using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    input_type = node->get_input_element_type(0);
    output_type = node->get_output_element_type(0);

    if (output_type == ov::element::bf16)
       uni_vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(host, host_isa));
}

void jit_convert_emitter::validate_types() const {
    auto is_supported_type = [this](const ov::element::Type& type) {
        return any_of(supported_types.begin(), supported_types.end(),
                      [&type](const ov::element::Type& supported_type) { return supported_type == type; } );
    };

    if (!is_supported_type(input_type))
        OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
    if (!is_supported_type(output_type))
        OV_CPU_JIT_EMITTER_THROW("Unsupported output type: ", output_type.get_type_name());
}

size_t jit_convert_emitter::get_inputs_num() const { return 1; }

void jit_convert_emitter::emit_data() const {
    jit_emitter::emit_data();
    if (uni_vcvtneps2bf16)
        uni_vcvtneps2bf16->emit_data();
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_emitter::float2bfloat(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst  = Vmm(out_vec_idxs[0]);
    if (!uni_vcvtneps2bf16)
        OV_CPU_JIT_EMITTER_THROW("Converter from float to bf16 isn't initialized!");

    uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, node, exec_prc) {
    prepare_table();
}

bool jit_convert_truncation_emitter::is_i8_and_u8_case() const {
    return one_of(input_type, ov::element::i8, ov::element::u8) &&
           one_of(output_type, ov::element::i8, ov::element::u8);
}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    validate_types();
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        emit_isa<cpu::x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst  = Vmm(out_vec_idxs[0]);

    Xmm xmm_dst = Xmm(out_vec_idxs[0]);
    Ymm ymm_dst = Ymm(out_vec_idxs[0]);

    // For Truncation behavior we can just move data from src to dst if we want convert i8 -> u8 or u8 -> i8
    if ((input_type == output_type) || is_i8_and_u8_case()) {
        if (vmm_src != vmm_dst) {
            h->uni_vmovups(vmm_dst, vmm_src);
        }
        return;
    }

    switch (input_type) {
        case ov::element::f32:
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvttps2dq(vmm_dst, vmm_src);
            break;
        case ov::element::i32:
            if (one_of(output_type, ov::element::f32, ov::element::bf16, ov::element::f16))
                h->uni_vcvtdq2ps(vmm_dst, vmm_src);
            break;
        case ov::element::bf16:
            h->vpmovzxwd(vmm_dst, vmm_src);
            h->uni_vpslld(vmm_dst, vmm_dst, 16);
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvttps2dq(vmm_dst, vmm_dst);
            break;
        case ov::element::f16:
            if (isa == dnnl::impl::cpu::x64::avx512_core)
                h->vcvtph2ps(vmm_dst, Ymm(vmm_src.getIdx()));
            else
                h->vcvtph2ps(vmm_dst,
                             Xmm(vmm_src.getIdx()));  // for avx2_vnni_2?
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvttps2dq(vmm_dst, vmm_dst);
            break;
        case ov::element::i8:
            h->uni_vpmovsxbd(vmm_dst, vmm_src);
            break;
        case ov::element::u8:
            h->uni_vpmovzxbd(vmm_dst, vmm_src);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported input data type");
    }

    switch (output_type) {
        case ov::element::f32:
            if (!one_of(input_type, ov::element::i32, ov::element::bf16, ov::element::f16)) {
                h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i32:
            break;
        case ov::element::bf16:
            if (input_type == ov::element::f32) {
                float2bfloat<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            } else {
                if (one_of(input_type, ov::element::i8, ov::element::u8)) {
                    h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
                float2bfloat<isa>({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            }
            break;
        case ov::element::f16:
            if (input_type == ov::element::f32) {
                if (isa == dnnl::impl::cpu::x64::avx512_core)
                    h->vcvtps2ph(ymm_dst, vmm_src, 0x4);
                else
                    h->vcvtps2ph(xmm_dst, vmm_src, 0x4);
            } else {
                if (one_of(input_type, ov::element::i8, ov::element::u8)) {
                    h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
                if (isa == dnnl::impl::cpu::x64::avx512_core)
                    h->vcvtps2ph(ymm_dst, vmm_dst, 0x4);
                else
                    h->vcvtps2ph(xmm_dst, vmm_dst, 0x4);
            }
            break;
        case ov::element::i8:
        case ov::element::u8:
            if (input_type == ov::element::i32) {
                dword2int8<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            } else {
                dword2int8<isa>({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            }
            break;
        default:
           OV_CPU_JIT_EMITTER_THROW("Unsupported output data type");
    }
}

void jit_convert_truncation_emitter::register_table_entries() {
    if (host_isa_ == dnnl::impl::cpu::x64::avx2 &&
        one_of(output_type, ov::element::i8, ov::element::u8) &&
        !is_i8_and_u8_case())
        push_arg_entry_of("mask_byte", 0x000000ff, true);
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_truncation_emitter::dword2int8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);

    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Xmm xmm_dst = Xmm(out_vec_idxs[0]);
    Ymm ymm_dst = Ymm(out_vec_idxs[0]);

    if (isa == dnnl::impl::cpu::x64::avx512_core) {
        h->vpmovdb(xmm_dst, vmm_src);
    } else if (isa == dnnl::impl::cpu::x64::avx2) {
        h->vpand(vmm_dst, vmm_src, table_val("mask_byte"));  // to avoid saturation
        h->uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
        if (isa != dnnl::impl::cpu::x64::sse41)
            h->vpermq(ymm_dst, ymm_dst, 0x08);
        h->uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
    }
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_saturation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    validate_types();
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        emit_isa<cpu::x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_saturation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst  = Vmm(out_vec_idxs[0]);

    Xmm xmm_dst = Xmm(out_vec_idxs[0]);
    Ymm ymm_dst = Ymm(out_vec_idxs[0]);

    if (input_type == output_type) {
        h->uni_vmovups(vmm_dst, vmm_src);
        return;
    }

    switch (input_type) {
        case ov::element::f32:
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvtps2dq(vmm_dst, vmm_src);
            break;
        case ov::element::i32:
            if (one_of(output_type, ov::element::f32, ov::element::bf16, ov::element::f16))
                h->uni_vcvtdq2ps(vmm_dst, vmm_src);
            break;
        case ov::element::bf16:
            h->vpmovzxwd(vmm_dst, vmm_src);
            h->uni_vpslld(vmm_dst, vmm_dst, 16);
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvttps2dq(vmm_dst, vmm_dst);
            break;
        case ov::element::f16:
            if (isa == dnnl::impl::cpu::x64::avx512_core)
                h->vcvtph2ps(vmm_dst, Ymm(vmm_src.getIdx()));
            else
                h->vcvtph2ps(vmm_dst,
                             Xmm(vmm_src.getIdx()));  // for avx2_vnni_2?
            if (one_of(output_type, ov::element::i32, ov::element::i8, ov::element::u8))
                h->uni_vcvttps2dq(vmm_dst, vmm_dst);
            break;
        case ov::element::i8:
            h->uni_vpmovsxbd(vmm_dst, vmm_src);
            break;
        case ov::element::u8:
            h->uni_vpmovzxbd(vmm_dst, vmm_src);
            break;
        default:
           OV_CPU_JIT_EMITTER_THROW("Unsupported input data type");
    }

    switch (output_type) {
        case ov::element::f32:
            if (!one_of(input_type, ov::element::i32, ov::element::bf16, ov::element::f16)) {
                h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i32:
            break;
        case ov::element::bf16:
            if (input_type == ov::element::f32) {
                float2bfloat<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            } else {
                if (one_of(input_type, ov::element::i8, ov::element::u8)) {
                    h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
                float2bfloat<isa>({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())});
            }
            break;
        case ov::element::f16:
            if (input_type == ov::element::f32) {
                if (isa == dnnl::impl::cpu::x64::avx512_core)
                    h->vcvtps2ph(ymm_dst, vmm_src, 0x4);
                else
                    h->vcvtps2ph(xmm_dst, vmm_src, 0x4);
            } else {
                if (one_of(input_type, ov::element::i8, ov::element::u8)) {
                    h->uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
                if (isa == dnnl::impl::cpu::x64::avx512_core)
                    h->vcvtps2ph(ymm_dst, vmm_dst, 0x4);
                else
                    h->vcvtps2ph(xmm_dst, vmm_dst, 0x4);
            }
            break;
        case ov::element::i8:
        case ov::element::u8:
            if (input_type == ov::element::i32) {
                dword2int8<isa>({static_cast<size_t>(vmm_src.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())}, output_type.is_signed());
            } else {
                dword2int8<isa>({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())}, output_type.is_signed());
            }
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported output data type");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_convert_saturation_emitter::dword2int8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs, bool is_signed) const {
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);

    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Xmm xmm_dst = Xmm(out_vec_idxs[0]);
    Ymm ymm_dst = Ymm(out_vec_idxs[0]);

    if (isa == dnnl::impl::cpu::x64::avx512_core) {
        if (is_signed) {
            h->vpmovsdb(xmm_dst, vmm_src);
        } else {
            Vmm vmm_zero  = Vmm(aux_vec_idxs[0]);
            h->vpxord(vmm_zero, vmm_zero, vmm_zero);
            h->vpmaxsd(vmm_dst, vmm_src, vmm_zero);
            h->vpmovusdb(xmm_dst, vmm_dst);
        }
    } else {
        if (is_signed)
            h->uni_vpackssdw(vmm_dst, vmm_src, vmm_src);
        else
            h->uni_vpackusdw(vmm_dst, vmm_src, vmm_src);

        if (isa != dnnl::impl::cpu::x64::sse41)
            h->vpermq(ymm_dst, ymm_dst, 0x08);

        if (is_signed)
            h->uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
        else
            h->uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
    }
}

size_t jit_convert_saturation_emitter::aux_vecs_count() const {
    // 1 register is for dword2int8 unsigned
    return output_type == ov::element::u8 && host_isa_ == dnnl::impl::cpu::x64::avx512_core? 1 : 0;
}

}   // namespace intel_cpu
}   // namespace ov
