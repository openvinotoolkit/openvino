// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak;

#define CONST_1_F 0x3f800000 // 1.f
#define INF_MASK  0x7F800000
#define INF_NEG_MASK 0xFF800000

namespace ov {
namespace intel_cpu {

namespace {
ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(input.get_source_output().get_element_type());
    }

    OV_CPU_JIT_EMITTER_ASSERT(std::all_of(input_precisions.cbegin(), input_precisions.cend(),
                                          [&input_precisions](const ov::element::Type& precision) { return precision == input_precisions[0]; }),
                           "Arithmetic binary node has not equal input precisions");

    return input_precisions[0];
}
}  // namespace

/// ADD ///
jit_add_emitter::jit_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_add_emitter::jit_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_add_emitter::get_inputs_num() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vadd = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: h->uni_vaddps(vmm_dst, vmm_src0, vmm_src1); break;
            case ov::element::i32:  h->uni_vpaddd(vmm_dst, vmm_src0, vmm_src1); break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vadd(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vadd(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_mul_add_emitter::jit_mul_add_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_mul_add_emitter::get_inputs_num() const { return 3; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_src2 = Vmm(in_vec_idxs[2]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vfmadd231_xmm = [this](Xmm vmm_dst, Xmm vmm_src0, Xmm vmm_src1, Xmm vmm_src2) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        switch (exec_prc_) {
            case ov::element::f32: {
                h->uni_vmulps(vmm_dst, vmm_dst, vmm_src1);
                h->uni_vaddps(vmm_dst, vmm_dst, vmm_src2);
            } break;
            case ov::element::i32: {
                h->uni_vpmulld(vmm_dst, vmm_dst, vmm_src1);
                h->uni_vpaddd(vmm_dst, vmm_dst, vmm_src2);
            } break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    auto uni_vfmadd231_vmm = [this, vmm_aux0](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1, Vmm vmm_src2) {
        switch (exec_prc_) {
            case ov::element::f32: {
                Vmm vmm_mul0;
                if (vmm_dst.getIdx() == vmm_src0.getIdx()) {
                    h->uni_vmovups(vmm_aux0, vmm_src0);
                    vmm_mul0 = vmm_aux0;
                } else {
                    vmm_mul0 = vmm_src0;
                }

                Vmm vmm_mul1;
                if (vmm_dst.getIdx() == vmm_src1.getIdx()) {
                    h->uni_vmovups(vmm_aux0, vmm_src1);
                    vmm_mul1 = vmm_aux0;
                } else {
                    vmm_mul1 = vmm_src1;
                }

                if (vmm_dst.getIdx() != vmm_src2.getIdx())
                    h->uni_vmovups(vmm_dst, vmm_src2);

                h->uni_vfmadd231ps(vmm_dst, vmm_mul0, vmm_mul1);
            } break;
            case ov::element::i32: {
                h->uni_vpmulld(vmm_dst, vmm_src0, vmm_src1);
                h->uni_vpaddd(vmm_dst, vmm_dst, vmm_src2);
            } break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        uni_vfmadd231_xmm(vmm_dst, vmm_src0, vmm_src1, vmm_src2);
    } else {
        uni_vfmadd231_vmm(vmm_dst, vmm_src0, vmm_src1, vmm_src2);
    }
}

size_t jit_mul_add_emitter::aux_vecs_count() const {
    return 1;
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}, {element::i32, element::i32, element::i32}};
}

/// SUB ///
jit_subtract_emitter::jit_subtract_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_subtract_emitter::jit_subtract_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_subtract_emitter::get_inputs_num() const { return 2; }

void jit_subtract_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vsub = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1); break;
            case ov::element::i32:  h->uni_vpsubd(vmm_dst, vmm_src0, vmm_src1); break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vsub(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vsub(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_multiply_emitter::jit_multiply_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_num() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vmul = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: h->uni_vmulps(vmm_dst, vmm_src0, vmm_src1); break;
            case ov::element::i32:  h->uni_vpmulld(vmm_dst, vmm_src0, vmm_src1); break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vmul(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vmul(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// DIVIDE ///
jit_divide_emitter::jit_divide_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_divide_emitter::jit_divide_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_num() const { return 2; }

void jit_divide_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vdiv = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: {
                h->uni_vdivps(vmm_dst, vmm_src0, vmm_src1);
                break;
            }
            case ov::element::i32: {
                Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);

                // The opset doesn't contain vector instruction for integer divide operation
                // As WA we emulate its behavior via fp divide followed by rounding to zero
                h->uni_vcvtdq2ps(vmm_dst, vmm_src0);
                h->uni_vcvtdq2ps(vmm_aux0, vmm_src1);
                h->uni_vdivps(vmm_dst, vmm_dst, vmm_aux0);
                h->uni_vroundps(vmm_dst, vmm_dst, 3); // rounding to zero
                h->uni_vcvtps2dq(vmm_dst, vmm_dst);
                break;
            }
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vdiv(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vdiv(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

size_t jit_divide_emitter::aux_vecs_count() const {
    return exec_prc_ == ov::element::i32 ? 1 : 0;
}

/// FLOOR ///
jit_floor_emitter::jit_floor_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}
jit_floor_emitter::jit_floor_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_floor_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_floor_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_floor_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_floor_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    h->uni_vroundps(vmm_dst, vmm_src, 1);
}

/// CEILING ///
jit_ceiling_emitter::jit_ceiling_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}
jit_ceiling_emitter::jit_ceiling_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_ceiling_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_ceiling_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_ceiling_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_ceiling_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    h->uni_vroundps(vmm_dst, vmm_src, 2);
}

/// FLOOR_MOD ///
jit_floor_mod_emitter::jit_floor_mod_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             const std::shared_ptr<ov::Node>& node,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}
jit_floor_mod_emitter::jit_floor_mod_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_floor_mod_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_floor_mod_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_floor_mod_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_floor_mod_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);

    if (isa == x64::sse41) {
        if (vmm_dst.getIdx() != vmm_src0.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        h->uni_vmovups(vmm_aux0, vmm_src0);
        h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vroundps(vmm_aux0, vmm_aux0, 1); // rounding down
        h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vsubps(vmm_dst, vmm_dst, vmm_aux0);
    } else {
        if (vmm_dst.getIdx() != vmm_src0.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        h->uni_vdivps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vroundps(vmm_aux0, vmm_aux0, 1); // rounding down
        h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vsubps(vmm_dst, vmm_dst, vmm_aux0);
    }
}

size_t jit_floor_mod_emitter::aux_vecs_count() const {
    return 1;
}

/// MOD ///
jit_mod_emitter::jit_mod_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}
jit_mod_emitter::jit_mod_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_mod_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_mod_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_mod_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_mod_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);

    if (isa == x64::sse41) {
        if (vmm_dst.getIdx() != vmm_src0.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        h->uni_vmovups(vmm_aux0, vmm_src0);
        h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vroundps(vmm_aux0, vmm_aux0, 3); // truncate
        h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vsubps(vmm_dst, vmm_dst, vmm_aux0);
    } else {
        if (vmm_dst.getIdx() != vmm_src0.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        h->uni_vdivps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vroundps(vmm_aux0, vmm_aux0, 3); // truncate
        h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_src1);
        h->uni_vsubps(vmm_dst, vmm_dst, vmm_aux0);
    }
}

size_t jit_mod_emitter::aux_vecs_count() const {
    return 1;
}

/// MAXIMUM ///
jit_maximum_emitter::jit_maximum_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_maximum_emitter::jit_maximum_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_maximum_emitter::get_inputs_num() const { return 2; }

void jit_maximum_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_maximum_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vmax = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: h->uni_vmaxps(vmm_dst, vmm_src0, vmm_src1); break;
            case ov::element::i32:  h->uni_vpmaxsd(vmm_dst, vmm_src0, vmm_src1); break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        if (vmm_src0.getIdx() != vmm_dst.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vmax(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vmax(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_maximum_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// MINIMUM ///
jit_minimum_emitter::jit_minimum_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}
jit_minimum_emitter::jit_minimum_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_minimum_emitter::get_inputs_num() const { return 2; }

void jit_minimum_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_minimum_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vmin = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: h->uni_vminps(vmm_dst, vmm_src0, vmm_src1); break;
            case ov::element::i32:  h->uni_vpminsd(vmm_dst, vmm_src0, vmm_src1); break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        if (vmm_src0.getIdx() != vmm_dst.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vmin(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vmin(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_minimum_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// SQUARED_DIFFERENCE ///
jit_squared_difference_emitter::jit_squared_difference_emitter(
    x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}
jit_squared_difference_emitter::jit_squared_difference_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_squared_difference_emitter::get_inputs_num() const { return 2; }

void jit_squared_difference_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_squared_difference_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    auto uni_vsqdiff = [this](Vmm vmm_dst, Vmm vmm_src0, Vmm vmm_src1) {
        switch (exec_prc_) {
            case ov::element::f32: {
                h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);
                h->uni_vmulps(vmm_dst, vmm_dst, vmm_dst);
            } break;
            case ov::element::i32: {
                h->uni_vpsubd(vmm_dst, vmm_src0, vmm_src1);
                h->uni_vpmulld(vmm_dst, vmm_dst, vmm_dst);
            } break;
            default: OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
        }
    };

    if (isa == x64::sse41) {
        if (vmm_src0.getIdx() != vmm_dst.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
        uni_vsqdiff(vmm_dst, vmm_dst, vmm_src1);
    } else {
        uni_vsqdiff(vmm_dst, vmm_src0, vmm_src1);
    }
}

std::set<std::vector<element::Type>> jit_squared_difference_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// POWER_DYNAMIC ///
jit_power_dynamic_emitter::jit_power_dynamic_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node,
                                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}
jit_power_dynamic_emitter::jit_power_dynamic_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_power_dynamic_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_power_dynamic_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_power_dynamic_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_power_dynamic_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);

    // caller obligation to save gprs as callee may use them
    size_t gpr_size = 8;
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                     h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    if (isa == x64::avx512_core) {
        h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
        for (size_t i = 0; i < n_k_regs_to_save; ++i) {
            if (x64::mayiuse(x64::avx512_core))
                h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
            else
                h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
        }
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. Additionally save space for vmm_src, to put the answer in-place on
    // this space and space for beta.
    // 3. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, (get_max_vecs_count() + 2) * get_vec_length());
    for (size_t i = 2; i < get_max_vecs_count() + 2; ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Vmm(i - 2));
    h->uni_vmovups(h->ptr[h->rsp + 0 * get_vec_length()], vmm_src0); // src
    h->uni_vmovups(h->ptr[h->rsp + 1 * get_vec_length()], vmm_src1); // beta

    // save function address in gpr to pass in in call instruction
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

    // align stack on 16-byte as ABI requires
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    // Take src, apply powf on it and replace value on a stack with dst.
    for (size_t i = 0; i < get_vec_length() / sizeof(float); ++i) {
        const Address &source = h->ptr[h->rsp + h->rbx + i * sizeof(float)];
        h->uni_vmovss(xmm0, source);
        h->uni_vmovss(xmm1, h->ptr[h->rsp + h->rbx + get_vec_length() + i * sizeof(float)]);
        h->call(h->rbp);
        h->uni_vmovss(source, xmm0);
    }

    h->add(h->rsp, h->rbx);

    // restore vector registers
    for (size_t i = get_max_vecs_count() + 1; i >= 2; --i)
        h->uni_vmovups(Vmm(i - 2), h->ptr[h->rsp + i * get_vec_length()]);
    h->uni_vmovups(vmm_dst, h->ptr[h->rsp + 0 * get_vec_length()]);
    h->add(h->rsp, (get_max_vecs_count() + 2) * get_vec_length());

    // restore k registers
    if (isa == x64::avx512_core) {
        for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
            if (x64::mayiuse(x64::avx512_core))
                h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            else
                h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        }
        h->add(h->rsp, n_k_regs_to_save * k_mask_size);
    }

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}


/// EQUAL ///
jit_equal_emitter::jit_equal_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_equal_emitter::jit_equal_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_equal_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_eq_oq);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->uni_vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_eq_oq);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_equal_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_equal_emitter::aux_vecs_count() const {
    return 2;
}

/// NOT_EQUAL ///
jit_not_equal_emitter::jit_not_equal_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             const std::shared_ptr<ov::Node>& node,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_not_equal_emitter::jit_not_equal_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_not_equal_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_not_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_not_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_not_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_eq_oq);
        h->movups(vmm_dst, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("one"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("zero"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_eq_oq);
        h->uni_vmovups(vmm_dst, table_val("one"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("zero"));
    }
}

void jit_not_equal_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_not_equal_emitter::aux_vecs_count() const {
    return 2;
}

/// GREATER ///
jit_greater_emitter::jit_greater_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_greater_emitter::jit_greater_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_greater_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_greater_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_greater_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_greater_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_gt_os);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpgtps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_gt_os);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_greater_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_greater_emitter::aux_vecs_count() const {
    return 2;
}

/// GREATER_EQUAL ///
jit_greater_equal_emitter::jit_greater_equal_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node,
                                                     ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_greater_equal_emitter::jit_greater_equal_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_greater_equal_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_greater_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_greater_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_greater_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_ge_os);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpgeps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_ge_os);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_greater_equal_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_greater_equal_emitter::aux_vecs_count() const {
    return 2;
}

/// LESS ///
jit_less_emitter::jit_less_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_less_emitter::jit_less_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_less_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_less_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_less_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_less_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_lt_os);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpltps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_lt_os);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_less_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_less_emitter::aux_vecs_count() const {
    return 2;
}

/// LESS_EQUAL ///
jit_less_equal_emitter::jit_less_equal_emitter(x64::jit_generator* host,
                                               x64::cpu_isa_t host_isa,
                                               const std::shared_ptr<ov::Node>& node,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_less_equal_emitter::jit_less_equal_emitter(x64::jit_generator* host,
                                               x64::cpu_isa_t host_isa,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_less_equal_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_less_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_less_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_less_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->movups(vmm_aux0, vmm_src0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_le_os);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpleps(vmm_aux0, vmm_src0, vmm_src1);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, vmm_src1, _cmp_le_os);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_less_equal_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_less_equal_emitter::aux_vecs_count() const {
    return 2;
}

/// LOGICAL_AND ///
jit_logical_and_emitter::jit_logical_and_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_logical_and_emitter::jit_logical_and_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_and_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_logical_and_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_logical_and_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_logical_and_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);
    Vmm vmm_aux2 = Vmm(aux_vec_idxs[2]);

    if (isa == x64::sse41) {
        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src0, _cmp_eq_oq);
        h->movups(vmm_dst, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_dst, vmm_aux1);

        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_eq_oq);
        h->movups(vmm_aux2, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_aux2, vmm_aux1);

        h->uni_vandps(vmm_dst, vmm_dst, vmm_aux2);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, table_val("zero"));
        h->uni_vmovups(vmm_dst, table_val("one"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("zero"), vmm_aux0);

        h->vcmpeqps(vmm_aux1, vmm_src1, table_val("zero"));
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendvps(vmm_aux0, vmm_aux0, table_val("zero"), vmm_aux1);

        h->uni_vandps(vmm_dst, vmm_dst, vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, table_val("zero"), _cmp_eq_oq);
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendmps(vmm_dst | k_mask, vmm_aux0, table_val("zero"));

        h->vcmpps(k_mask, vmm_src1, table_val("zero"), _cmp_eq_oq);
        h->vblendmps(vmm_aux0 | k_mask, vmm_aux0, table_val("zero"));

        h->uni_vandps(vmm_dst, vmm_dst, vmm_aux0);
    }
}

void jit_logical_and_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_logical_and_emitter::aux_vecs_count() const {
    return 3;
}

/// LOGICAL_OR ///
jit_logical_or_emitter::jit_logical_or_emitter(x64::jit_generator* host,
                                               x64::cpu_isa_t host_isa,
                                               const std::shared_ptr<ov::Node>& node,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_logical_or_emitter::jit_logical_or_emitter(x64::jit_generator* host,
                                               x64::cpu_isa_t host_isa,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_or_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_logical_or_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_logical_or_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_logical_or_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);
    Vmm vmm_aux2 = Vmm(aux_vec_idxs[2]);

    if (isa == x64::sse41) {
        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src0, _cmp_eq_oq);
        h->movups(vmm_dst, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_dst, vmm_aux1);

        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_eq_oq);
        h->movups(vmm_aux2, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_aux2, vmm_aux1);

        h->uni_vorps(vmm_dst, vmm_dst, vmm_aux2);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, table_val("zero"));
        h->uni_vmovups(vmm_dst, table_val("one"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("zero"), vmm_aux0);

        h->vcmpeqps(vmm_aux1, vmm_src1, table_val("zero"));
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendvps(vmm_aux0, vmm_aux0, table_val("zero"), vmm_aux1);

        h->uni_vorps(vmm_dst, vmm_dst, vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, table_val("zero"), _cmp_eq_oq);
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendmps(vmm_dst | k_mask, vmm_aux0, table_val("zero"));

        h->vcmpps(k_mask, vmm_src1, table_val("zero"), _cmp_eq_oq);
        h->vblendmps(vmm_aux0 | k_mask, vmm_aux0, table_val("zero"));

        h->uni_vorps(vmm_dst, vmm_dst, vmm_aux0);
    }
}

void jit_logical_or_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_logical_or_emitter::aux_vecs_count() const {
    return 3;
}

/// LOGICAL_XOR ///
jit_logical_xor_emitter::jit_logical_xor_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_logical_xor_emitter::jit_logical_xor_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_xor_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_logical_xor_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_logical_xor_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_logical_xor_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);
    Vmm vmm_aux2 = Vmm(aux_vec_idxs[2]);

    if (isa == x64::sse41) {
        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src0, _cmp_eq_oq);
        h->movups(vmm_dst, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_dst, vmm_aux1);

        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src1, _cmp_eq_oq);
        h->movups(vmm_aux2, table_val("one"));
        h->pxor(vmm_aux1, vmm_aux1);
        h->blendvps(vmm_aux2, vmm_aux1);

        h->uni_vxorps(vmm_dst, vmm_dst, vmm_aux2);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, table_val("zero"));
        h->uni_vmovups(vmm_dst, table_val("one"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("zero"), vmm_aux0);

        h->vcmpeqps(vmm_aux1, vmm_src1, table_val("zero"));
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendvps(vmm_aux0, vmm_aux0, table_val("zero"), vmm_aux1);

        h->uni_vxorps(vmm_dst, vmm_dst, vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, table_val("zero"), _cmp_eq_oq);
        h->uni_vmovups(vmm_aux0, table_val("one"));
        h->vblendmps(vmm_dst | k_mask, vmm_aux0, table_val("zero"));

        h->vcmpps(k_mask, vmm_src1, table_val("zero"), _cmp_eq_oq);
        h->vblendmps(vmm_aux0 | k_mask, vmm_aux0, table_val("zero"));

        h->uni_vxorps(vmm_dst, vmm_dst, vmm_aux0);
    }
}

void jit_logical_xor_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_logical_xor_emitter::aux_vecs_count() const {
    return 3;
}

/// LOGICAL_NOT ///
jit_logical_not_emitter::jit_logical_not_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_logical_not_emitter::jit_logical_not_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_not_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_logical_not_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_logical_not_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_logical_not_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src0, _cmp_eq_oq);
        h->movups(vmm_aux1, table_val("one"));
        h->pxor(vmm_dst, vmm_dst);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vcmpeqps(vmm_aux0, vmm_src0, table_val("zero"));
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendvps(vmm_dst, vmm_dst, table_val("one"), vmm_aux0);
    } else {
        h->vcmpps(k_mask, vmm_src0, table_val("zero"), _cmp_eq_oq);
        h->uni_vmovups(vmm_dst, table_val("zero"));
        h->vblendmps(vmm_dst | k_mask, vmm_dst, table_val("one"));
    }
}

void jit_logical_not_emitter::register_table_entries() {
    push_arg_entry_of("zero", 0x00000000, true);
    push_arg_entry_of("one", CONST_1_F, true);
}

size_t jit_logical_not_emitter::aux_vecs_count() const {
    return 2;
}

/// POWER_STATIC ///
jit_power_static_emitter::jit_power_static_emitter(x64::jit_generator* host,
                                                   x64::cpu_isa_t host_isa,
                                                   const std::shared_ptr<ov::Node>& node,
                                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to snippets::op::PowerStatic");
    }

    power = powerStaticNode->get_power();
    scale = 1.f;
    shift = 0.f;

    prepare_table();
}

jit_power_static_emitter::jit_power_static_emitter(x64::jit_generator *host, x64::cpu_isa_t host_isa,
                                                   float inpPower, float inpScale, float inpShift,
                                                   ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc), power(inpPower), scale(inpScale), shift(inpShift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_power_static_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);

    Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);

    if (scale != 1.f || shift != 0.f) {
        if (isa == x64::sse41) {
            h->uni_vmovups(vmm_aux0, table_val("scale"));
            h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_src0);
            h->uni_vmovups(vmm_dst, table_val("shift"));
            h->uni_vaddps(vmm_dst, vmm_dst, vmm_aux0);
        } else {
            if (vmm_dst.getIdx() != vmm_src0.getIdx()) {
                h->uni_vmovups(vmm_dst, table_val("shift"));
                h->uni_vfmadd231ps(vmm_dst, vmm_src0, table_val("scale"));
            } else {
                h->uni_vmovups(vmm_aux0, table_val("shift"));
                h->uni_vfmadd231ps(vmm_aux0, vmm_src0, table_val("scale"));
                h->uni_vmovups(vmm_dst, vmm_aux0);
            }
        }
    } else {
        if (vmm_dst.getIdx() != vmm_src0.getIdx())
            h->uni_vmovups(vmm_dst, vmm_src0);
    }

    if (power == 1.f) {
    } else if (power == 0.5f || power == -0.5f) {
        h->uni_vsqrtps(vmm_dst, vmm_dst);

        if (power < 0.f) {
            h->uni_vmovups(vmm_aux0, table_val("one"));
            if (isa == x64::sse41) {
                h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_dst);
                h->uni_vmovups(vmm_dst, vmm_aux0);
            } else {
                h->uni_vdivps(vmm_dst, vmm_aux0, vmm_dst);
            }
        }
    } else if (std::floor(power) == power && power != 0) {
        int64_t ipower = std::abs(static_cast<int64_t>(power)) - 1;
        h->uni_vmovups(vmm_aux0, vmm_dst);
        while (ipower > 0) {
            if (ipower & 0x1)
                h->uni_vmulps(vmm_dst, vmm_dst, vmm_aux0);
            if (ipower > 1)
                h->uni_vmulps(vmm_aux0, vmm_aux0, vmm_aux0);
            ipower = ipower >> 1;
        }

        if (power < 0.f) {
            h->uni_vmovups(vmm_aux0, table_val("one"));
            if (isa == x64::sse41) {
                h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_dst);
                h->uni_vmovups(vmm_dst, vmm_aux0);
            } else {
                h->uni_vdivps(vmm_dst, vmm_aux0, vmm_dst);
            }
        }
    } else {
        h->uni_vmovups(vmm_aux0, table_val("power"));

        // caller obligation to save gprs as callee may use them
        size_t gpr_size = 8;
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // caller obligation to save k-regs as callee may use them
        size_t n_k_regs_to_save = 8;
        if (isa == x64::avx512_core) {
            h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
            for (size_t i = 0; i < n_k_regs_to_save; ++i) {
                if (x64::mayiuse(x64::avx512_core))
                    h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
                else
                    h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
            }
        }

        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. Additionally save space for vmm_src, to put the answer in-place on
        // this space and space for beta.
        // 3. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        h->sub(h->rsp, (get_max_vecs_count() + 2) * get_vec_length());
        for (size_t i = 2; i < get_max_vecs_count() + 2; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Vmm(i - 2));
        h->uni_vmovups(h->ptr[h->rsp + 0 * get_vec_length()], vmm_dst); // src
        h->uni_vmovups(h->ptr[h->rsp + 1 * get_vec_length()], vmm_aux0); // beta

        // save function address in gpr to pass in in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

        // align stack on 16-byte as ABI requires
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        // Take src, apply powf on it and replace value on a stack with dst.
        for (size_t i = 0; i < get_vec_length() / sizeof(float); ++i) {
            const Address &source = h->ptr[h->rsp + h->rbx + i * sizeof(float)];
            h->uni_vmovss(xmm0, source);
            h->uni_vmovss(xmm1, h->ptr[h->rsp + h->rbx + get_vec_length() + i * sizeof(float)]);
            h->call(h->rbp);
            h->uni_vmovss(source, xmm0);
        }

        h->add(h->rsp, h->rbx);

        // restore vector registers
        for (size_t i = get_max_vecs_count() + 1; i >= 2; --i)
            h->uni_vmovups(Vmm(i - 2), h->ptr[h->rsp + i * get_vec_length()]);
        h->uni_vmovups(vmm_dst, h->ptr[h->rsp + 0 * get_vec_length()]);
        h->add(h->rsp, (get_max_vecs_count() + 2) * get_vec_length());

        // restore k registers
        if (isa == x64::avx512_core) {
            for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
                if (x64::mayiuse(x64::avx512_core))
                    h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
                else
                    h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            }
            h->add(h->rsp, n_k_regs_to_save * k_mask_size);
        }

        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }
}

void jit_power_static_emitter::register_table_entries() {
    push_arg_entry_of("power", x64::float2int(power), true);
    push_arg_entry_of("scale", x64::float2int(scale), true);
    push_arg_entry_of("shift", x64::float2int(shift), true);
    push_arg_entry_of("one",   x64::float2int(1.f), true);
}

size_t jit_power_static_emitter::aux_vecs_count() const {
    return 1;
}

/// PRELU ///
jit_prelu_emitter::jit_prelu_emitter(x64::jit_generator* host,
                                     x64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_prelu_emitter::jit_prelu_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
size_t jit_prelu_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);

    if (isa == x64::sse41) {
        h->pxor(vmm_aux0, vmm_aux0);
        h->cmpps(vmm_aux0, vmm_src0, _cmp_gt_os);
        h->movups(vmm_aux1, vmm_src1);
        h->mulps(vmm_aux1, vmm_src0);
        if (vmm_src0.getIdx() != vmm_dst.getIdx())
            h->movups(vmm_dst, vmm_src0);
        h->blendvps(vmm_dst, vmm_aux1);
    } else if (isa == x64::avx2) {
        h->vmulps(vmm_aux0, vmm_src0, vmm_src1);
        h->vxorps(vmm_aux1, vmm_aux1, vmm_aux1);
        h->vcmpgtps(vmm_aux1, vmm_src0, vmm_aux1);
        h->vblendvps(vmm_dst, vmm_aux0, vmm_src0, vmm_aux1);
    } else if (isa == x64::avx512_core) {
        h->vxorpd(vmm_aux0, vmm_aux0, vmm_aux0);
        if (vmm_src0.getIdx() != vmm_dst.getIdx())
            h->vmovups(vmm_dst, vmm_src0);
        h->vcmpps(k_mask, vmm_src0, vmm_aux0, _cmp_lt_os);
        h->vmulps(vmm_dst | k_mask, vmm_src0, vmm_src1);
    }
}

size_t jit_prelu_emitter::aux_vecs_count() const {
    return 2;
}

/// SQRT ///
jit_sqrt_emitter::jit_sqrt_emitter(x64::jit_generator* host,
                                   x64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}
jit_sqrt_emitter::jit_sqrt_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_sqrt_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_sqrt_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_sqrt_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_sqrt_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

     h->uni_vsqrtps(vmm_dst, vmm_src0);
}

/// Negate ///
jit_negative_emitter::jit_negative_emitter(x64::jit_generator* host,
                                           x64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_negative_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_negative_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_negative_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_negative_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst  = Vmm(out_vec_idxs[0]);
    h->uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
    h->uni_vsubps(vmm_dst, vmm_dst, vmm_src);
}


/// EXP ///
jit_exp_emitter::jit_exp_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_exp_emitter::jit_exp_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_exp_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_exp_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_exp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    Vmm vmm_mask = need_vmm_mask() ? Vmm(aux_vec_idxs[0]) : Vmm();
    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0 + static_cast<size_t>(need_vmm_mask())]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1 + static_cast<size_t>(need_vmm_mask())]);

    auto compute_cmp_mask = [&](const Vmm &vmm_src, const Xbyak::Operand &compare_operand, int cmp_predicate) {
        if (host_isa_ == x64::avx512_core) {
            h->vcmpps(k_mask, vmm_src, compare_operand, cmp_predicate);
        } else {
            h->uni_vcmpps(vmm_mask, vmm_src, compare_operand, cmp_predicate);
        }
    };

    auto blend_with_mask = [&](const Vmm &vmm_dst, const Xbyak::Operand &src) {
        if (host_isa_ == x64::avx512_core) {
            h->vblendmps(vmm_dst | k_mask, vmm_dst, src);
        } else {
            h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask);
        }
    };

    h->uni_vmovups(vmm_aux1, table_val("ln_flt_min_f"));
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, vmm_aux1, _cmp_lt_os);

    h->uni_vminps(vmm_dst, vmm_src, table_val("ln_flt_max_f"));
    h->uni_vmaxps(vmm_dst, vmm_dst, vmm_aux1);
    h->uni_vmovups(vmm_aux0, vmm_dst);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_dst, vmm_dst, table_val("log2ef"));
    h->uni_vaddps(vmm_dst, vmm_dst, table_val("half"));

    // tmp = floorf(fx)
    const auto _op_floor = 1u;
    h->uni_vroundps(vmm_aux1, vmm_dst, _op_floor);

    // keep vmm_dst = fx for further computations
    h->uni_vmovups(vmm_dst, vmm_aux1);

    // x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux0, vmm_aux1, table_val("ln2f"));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux1, vmm_dst);
    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val("exponent_bias"));
    const int n_mantissa_bits = 23;
    h->uni_vpslld(vmm_aux1, vmm_aux1, n_mantissa_bits);

    // use vmm_dst as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux1, vmm_dst);

    // compute polynomial
    h->uni_vmovups(vmm_dst, table_val("pol5"));
    h->uni_vfmadd213ps(vmm_dst, vmm_aux0, table_val("pol4"));
    h->uni_vfmadd213ps(vmm_dst, vmm_aux0, table_val("pol3"));
    h->uni_vfmadd213ps(vmm_dst, vmm_aux0, table_val("pol2"));
    h->uni_vfmadd213ps(vmm_dst, vmm_aux0, table_val("pol1"));
    h->uni_vfmadd213ps(vmm_dst, vmm_aux0, table_val("one"));
    // y = y * 2^n
    h->uni_vmulps(vmm_dst, vmm_dst, vmm_aux1);
}

void jit_exp_emitter::register_table_entries() {
    push_arg_entry_of("pol1", 0x3f7ffffb, true); // p1 = 0.999999701f
    push_arg_entry_of("pol2", 0x3efffee3, true); // p2 = 0.499991506f
    push_arg_entry_of("pol3", 0x3e2aad40, true); // p3 = 0.166676521f
    push_arg_entry_of("pol4", 0x3d2b9d0d, true); // p4 = 0.0418978221f
    push_arg_entry_of("pol5", 0x3c07cfce, true); // p5 = 0.00828929059f

    push_arg_entry_of("one", CONST_1_F, true);
    push_arg_entry_of("half", 0x3f000000, true);
    push_arg_entry_of("ln2f", 0x3f317218, true);
    push_arg_entry_of("log2ef", 0x3fb8aa3b, true);
    push_arg_entry_of("ln_flt_max_f", 0x42b17218, true);
    push_arg_entry_of("ln_flt_min_f", 0xc2aeac50, true);
    push_arg_entry_of("exponent_bias", 0x0000007f, true);
}

size_t jit_exp_emitter::aux_vecs_count() const {
    return need_vmm_mask() ? 3 : 2;
}

/// ERF ///
jit_erf_emitter::jit_erf_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    m_exp_emitter.reset(new jit_exp_emitter(host, host_isa, exec_prc));
    prepare_table();
}

jit_erf_emitter::jit_erf_emitter(x64::jit_generator* host,  x64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_erf_emitter(host, host_isa, exec_prc) {}

size_t jit_erf_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_erf_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_erf_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_erf_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    Vmm vmm_aux0 = Vmm(aux_vec_idxs[0]);
    Vmm vmm_aux1 = Vmm(aux_vec_idxs[1]);
    Vmm vmm_aux2 = Vmm(aux_vec_idxs[2]);
    Vmm vmm_aux3 = Vmm(aux_vec_idxs[3]);

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);

    // -exp(-x*x)
    h->uni_vmulps(vmm_dst, vmm_src, vmm_src);
    h->uni_vxorps(vmm_dst, vmm_dst, table_val("sign_mask"));

    // pass the current `aux_vec_idxs` to `exp_emitter` excepting `vmm_aux3`
    auto exp_aux_vec_idxs = aux_vec_idxs;
    exp_aux_vec_idxs.erase(std::find(exp_aux_vec_idxs.begin(), exp_aux_vec_idxs.end(), static_cast<size_t>(vmm_aux3.getIdx())));
    m_exp_emitter->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(vmm_dst.getIdx())}, exp_aux_vec_idxs);

    h->uni_vxorps(vmm_dst, vmm_dst, table_val("sign_mask"));

    // get sign
    h->uni_vmovups(vmm_aux0, vmm_aux3);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val("sign_mask"));

    // abs(x)
    h->uni_vmovups(vmm_aux1, vmm_aux3);
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val("positive_mask"));

    // t = 1 / (p*x + 1)
    h->uni_vmovups(vmm_aux2, table_val("approx_const"));
    h->uni_vfmadd213ps(vmm_aux2, vmm_aux1, table_val("one"));
    h->uni_vmovups(vmm_aux3, table_val("one"));
    h->uni_vdivps(vmm_aux3, vmm_aux3, vmm_aux2);

    // -exp(-x*x)*t
    h->uni_vmulps(vmm_dst, vmm_dst, vmm_aux3);

    // compute polynomialial r
    h->uni_vmovups(vmm_aux1, table_val("pol5"));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val("pol4"));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val("pol3"));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val("pol2"));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val("pol1"));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_dst, vmm_aux1, table_val("one"));
    h->uni_vxorps(vmm_dst, vmm_dst, vmm_aux0);
}

void jit_erf_emitter::register_table_entries() {
    push_arg_entry_of("approx_const", 0x3ea7ba05, true); // 0.3275911
    push_arg_entry_of("one", CONST_1_F, true);
    push_arg_entry_of("sign_mask", 0x80000000, true);
    push_arg_entry_of("positive_mask", 0x7fffffff, true);

    push_arg_entry_of("pol1", 0x3e827906, true); // p1 = 0.254829592f
    push_arg_entry_of("pol2", 0xbe91a98e, true); // p2 = -0.284496736f
    push_arg_entry_of("pol3", 0x3fb5f0e3, true); // p3 = 1.421413741f
    push_arg_entry_of("pol4", 0xbfba00e3, true); // p4 = -1.453152027f
    push_arg_entry_of("pol5", 0x3f87dc22, true); // p5 = 1.061405429f
}

size_t jit_erf_emitter::aux_vecs_count() const {
    return 4ul;
}

void jit_erf_emitter::emit_data() const {
    jit_emitter::emit_data();
    m_exp_emitter->emit_data();
}

/// SOFT SIGN ///
jit_soft_sign_emitter::jit_soft_sign_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             const std::shared_ptr<ov::Node>& node,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_soft_sign_emitter::jit_soft_sign_emitter(x64::jit_generator* host,
                                             x64::cpu_isa_t host_isa,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_soft_sign_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_soft_sign_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_soft_sign_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_soft_sign_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    h->uni_vmovups(vmm_dst, vmm_src);                             // y = x
    h->uni_vandps(vmm_src, vmm_src, table_val("positive_mask"));  // x = abs(x)
    h->uni_vaddps(vmm_src, vmm_src, table_val("one"));            // x++
    h->uni_vdivps(vmm_dst, vmm_dst, vmm_src);                     // y = y/x
}

void jit_soft_sign_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F, true);
    push_arg_entry_of("positive_mask", 0x7fffffff, true);
}

/// IS_FINITE ///
template <>
void jit_is_finite_emitter::emit_isa<x64::avx512_core>(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    auto vmm_src = Zmm(in_vec_idxs[0]);
    auto vmm_dst = Zmm(out_vec_idxs[0]);
    auto &ones_mask = h->k1;
    auto reg32_one = Reg32(aux_gpr_idxs[0]);

    h->mov(reg32_one, CONST_1_F);
    h->vfpclassps(ones_mask, vmm_src, 0B10111001);
    h->knotd(ones_mask, ones_mask);
    h->vpbroadcastd(vmm_dst | ones_mask | h->T_z, reg32_one);
}

template <x64::cpu_isa_t isa>
void jit_is_finite_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional<isa == x64::sse41, Xmm, Ymm>::type;
    auto vmm_src = Vmm(in_vec_idxs[0]);
    auto vmm_dst = Vmm(out_vec_idxs[0]);

    h->uni_vandps(vmm_src, vmm_src, table_val("inf"));
    h->uni_vcmpps(vmm_src, vmm_src, table_val("inf"), 0B00000100); // NEq

    if (isa == x64::avx2) {
        h->uni_vandps(vmm_dst, vmm_src, table_val("one"));
    } else {
        if (vmm_src.getIdx() != vmm_dst.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src);
        }
        h->uni_vandps(vmm_dst, vmm_dst, table_val("one"));
    }
}

void jit_is_finite_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

void jit_is_finite_emitter::register_table_entries() {
    if (host_isa_ != x64::avx512_core) {
        push_arg_entry_of("one", CONST_1_F, true);
        push_arg_entry_of("inf", INF_MASK, true);
    }
}

/// IS_INF ///
template <>
void jit_is_inf_emitter::emit_isa<x64::avx512_core>(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    Zmm vmm_src = Zmm(in_vec_idxs[0]);
    Zmm vmm_dst = Zmm(out_vec_idxs[0]);

    if (detect_negative || detect_positive) {
        auto &ones_mask = h->k1;
        auto reg32_one = Reg32(aux_gpr_idxs[0]);
        uint8_t imm = detect_negative ? 0B00010000 : 0B00000000;
        if (detect_positive) {
            imm |= 0B00001000;
        }

        h->mov(reg32_one, CONST_1_F);
        h->vfpclassps(ones_mask, vmm_src, imm);
        h->vpbroadcastd(vmm_dst | ones_mask | h->T_z, reg32_one);
    } else {
        h->uni_vxorps(vmm_dst, vmm_dst, vmm_dst);
    }
}

template <x64::cpu_isa_t isa>
void jit_is_inf_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional<isa == x64::sse41, Xmm, Ymm>::type;

    if (detect_negative || detect_positive) {
        auto vmm_src = Vmm(in_vec_idxs[0]);
        auto vmm_dst = Vmm(out_vec_idxs[0]);

        if (detect_positive) {
            if (detect_negative) {
                h->uni_vpslld(vmm_src, vmm_src, 1);
                h->uni_vpsrld(vmm_src, vmm_src, 1);
            }
            h->uni_vpcmpeqd(vmm_src, vmm_src, table_val("inf"));
        } else if (detect_negative) {
            h->uni_vpcmpeqd(vmm_src, vmm_src, table_val("inf_neg"));
        }

        if (isa == x64::avx2) {
            h->uni_vandps(vmm_dst, vmm_src, table_val("one"));
        } else {
            if (vmm_src.getIdx() != vmm_dst.getIdx()) {
                h->uni_vmovups(vmm_dst, vmm_src);
            }
            h->uni_vandps(vmm_dst, vmm_dst, table_val("one"));
        }
    } else {
        auto vmm_dst = Vmm(out_vec_idxs[0]);
        h->uni_vxorps(vmm_dst, vmm_dst, vmm_dst);
    }
}

void jit_is_inf_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

void jit_is_inf_emitter::register_table_entries() {
    if (host_isa_ != x64::avx512_core) {
        push_arg_entry_of("one", CONST_1_F, true);
        push_arg_entry_of("inf", INF_MASK, true);
        push_arg_entry_of("inf_neg", INF_NEG_MASK, true);
    }
}

/// IS_NAN ///
template <>
void jit_is_nan_emitter::emit_isa<x64::avx512_core>(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    auto vmm_src = Zmm(in_vec_idxs[0]);
    auto vmm_dst = Zmm(out_vec_idxs[0]);
    auto &ones_mask = h->k1;
    auto reg32_one = Reg32(aux_gpr_idxs[0]);

    h->mov(reg32_one, CONST_1_F);
    h->vfpclassps(ones_mask, vmm_src, 0B10000001);
    h->vpbroadcastd(vmm_dst | ones_mask | h->T_z, reg32_one);
}

template <x64::cpu_isa_t isa>
void jit_is_nan_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional<isa == x64::sse41, Xmm, Ymm>::type;
    auto vmm_src = Vmm(in_vec_idxs[0]);
    auto vmm_dst = Vmm(out_vec_idxs[0]);

    h->uni_vcmpps(vmm_src, vmm_src, vmm_src, 0B00000100);
    if (isa == x64::avx2) {
        h->uni_vandps(vmm_dst, vmm_src, table_val("one"));
    } else {
        if (vmm_dst.getIdx() != vmm_src.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src);
        }
        h->uni_vandps(vmm_dst, vmm_dst, table_val("one"));
    }
}

void jit_is_nan_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

void jit_is_nan_emitter::register_table_entries() {
    if (host_isa_ != x64::avx512_core) {
        push_arg_entry_of("one", CONST_1_F, true);
    }
}

/// SELECT ///
jit_select_emitter::jit_select_emitter(x64::jit_generator* host,
                                       x64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node,
                                       ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}
jit_select_emitter::jit_select_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_select_emitter::get_inputs_num() const { return 3; }

std::set<std::vector<element::Type>> jit_select_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

size_t jit_select_emitter::aux_vecs_count() const {
    if (host_isa_ == x64::avx512_core)
        return 0;
    else if (host_isa_ == x64::avx2)  // tmp vec for mask
        return 1;
    else // mask should be xmm0 on sse41 +  tmp vec for mask
        return 2;
}

void jit_select_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_select_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_cond = Vmm(in_vec_idxs[0]);
    Vmm vmm_src0 = Vmm(in_vec_idxs[1]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[2]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    if (isa == x64::sse41) {
        Vmm vmm_mask = Vmm(aux_vec_idxs[0]);
        Vmm vmm_zero = Vmm(aux_vec_idxs[1]);
        h->uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        h->uni_vcmpps(vmm_cond, vmm_cond, vmm_zero, 0x4);
        if (vmm_mask.getIdx() != vmm_cond.getIdx()) {
            h->uni_vmovups(vmm_mask, vmm_cond);
        }
        if (vmm_src1.getIdx() != vmm_dst.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src1);
        }
        h->uni_vblendvps(vmm_dst, vmm_dst, vmm_src0, vmm_mask);
    } else if (isa == x64::avx2) {
        Vmm vmm_zero = Vmm(aux_vec_idxs[0]);
        h->uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        h->uni_vcmpps(vmm_cond, vmm_cond, vmm_zero, 0x4);
        h->uni_vblendvps(vmm_dst, vmm_src1, vmm_src0, vmm_cond);
    } else {
        h->vptestmd(k_mask, vmm_cond, vmm_cond);
        h->vblendmps(vmm_dst | k_mask, vmm_src1, vmm_src0);
    }
}

/// BITWISE_AND ///
jit_bitwise_and_emitter::jit_bitwise_and_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_bitwise_and_emitter::jit_bitwise_and_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_bitwise_and_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_bitwise_and_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {
        {element::i8, element::i8},
        {element::u8, element::u8},
        {element::i32, element::i32}
    };
}

void jit_bitwise_and_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_bitwise_and_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    if (isa == x64::sse41) {
        if (vmm_dst.getIdx() != vmm_src0.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }
        h->andps(vmm_dst, vmm_src1);
    } else if ((host_isa_ == x64::avx2) || (host_isa_ == x64::avx512_core)) {
        h->vandps(vmm_dst, vmm_src0, vmm_src1);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

/// BITWISE_NOT ///
jit_bitwise_not_emitter::jit_bitwise_not_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_bitwise_not_emitter::jit_bitwise_not_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_bitwise_not_emitter::get_inputs_num() const { return 1; }

std::set<std::vector<element::Type>> jit_bitwise_not_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {
        {element::i8},
        {element::u8},
        {element::i32}
    };
}

size_t jit_bitwise_not_emitter::aux_vecs_count() const { return 1; }

void jit_bitwise_not_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_bitwise_not_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src = Vmm(in_vec_idxs[0]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    if (isa == x64::sse41) {
        if (vmm_dst.getIdx() != vmm_src.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src);
        }
        h->andnps(vmm_dst, table_val("all_bits"));
    } else if ((host_isa_ == x64::avx2) || (host_isa_ == x64::avx512_core)) {
        h->vandnps(vmm_dst, vmm_src, table_val("all_bits"));
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

void jit_bitwise_not_emitter::register_table_entries() {
    push_arg_entry_of("all_bits", 0xFFFFFFFF, true);
}

/// BITWISE_OR ///
jit_bitwise_or_emitter::jit_bitwise_or_emitter(x64::jit_generator* host,
                                               x64::cpu_isa_t host_isa,
                                               const std::shared_ptr<ov::Node>& node,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_bitwise_or_emitter::jit_bitwise_or_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_bitwise_or_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_bitwise_or_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {
        {element::i8, element::i8},
        {element::u8, element::u8},
        {element::i32, element::i32}
    };
}

void jit_bitwise_or_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_bitwise_or_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    if (isa == x64::sse41) {
        if (vmm_dst.getIdx() != vmm_src0.getIdx()) {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }
        h->orps(vmm_dst, vmm_src1);
    } else if ((host_isa_ == x64::avx2) || (host_isa_ == x64::avx512_core)) {
        h->vorps(vmm_dst, vmm_src0, vmm_src1);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

/// BITWISE_XOR ///
jit_bitwise_xor_emitter::jit_bitwise_xor_emitter(x64::jit_generator* host,
                                                 x64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_bitwise_xor_emitter::jit_bitwise_xor_emitter(x64::jit_generator* host, x64::cpu_isa_t host_isa, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_bitwise_xor_emitter::get_inputs_num() const { return 2; }

std::set<std::vector<element::Type>> jit_bitwise_xor_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {
        {element::i8, element::i8},
        {element::u8, element::u8},
        {element::i32, element::i32}
    };
}

void jit_bitwise_xor_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == x64::sse41) {
        emit_isa<x64::sse41>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx2) {
        emit_isa<x64::avx2>(in_vec_idxs, out_vec_idxs);
    } else if (host_isa_ == x64::avx512_core) {
        emit_isa<x64::avx512_core>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <x64::cpu_isa_t isa>
void jit_bitwise_xor_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in_vec_idxs[0]);
    Vmm vmm_src1 = Vmm(in_vec_idxs[1]);
    Vmm vmm_dst = Vmm(out_vec_idxs[0]);

    h->uni_vxorps(vmm_dst, vmm_src0, vmm_src1);
}

}   // namespace intel_cpu
}   // namespace ov
