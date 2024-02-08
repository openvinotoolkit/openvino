// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <memory>
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

namespace {
ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(
            input.get_source_output().get_element_type());
    }

    assert(std::all_of(
        input_precisions.begin(),
        input_precisions.end(),
        [&input_precisions](const ov::element::Type& precision) {return precision == input_precisions[0]; }));

    return input_precisions[0];
}
} // namespace

/// ADD ///
jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
                                 : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_add_emitter::get_inputs_count() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if ((exec_prc_ != ov::element::f16) && (exec_prc_ != ov::element::f32)) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    switch (exec_prc_) {
        case element::f16: {
            h->uni_fadd(dst.h, src0.h, src1.h);
            break;
        }
        case element::f32: {
            h->uni_fadd(dst.s, src0.s, src1.s);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// DIVIDE ///
jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_count() const { return 2; }

void jit_divide_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    switch (exec_prc_) {
        case element::f16: {
            std::cout << "jit_divide_emitter::emit_isa: element::f16" << std::endl;
            h->uni_fdiv(dst.h, src0.h, src1.h);
            break;
        }
        case element::f32: {
            std::cout << "jit_divide_emitter::emit_isa: element::f32" << std::endl;
            h->uni_fdiv(dst.s, src0.s, src1.s);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}


/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
                                         : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc)
                                         : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_mul_add_emitter::get_inputs_count() const { return 3; }

size_t jit_mul_add_emitter::get_aux_vecs_count() const { return 1; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if ((exec_prc_ != element::f16) && (exec_prc_ != element::f32)) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg dst = TReg(out_vec_idxs[0]);

    TReg mul0(in_vec_idxs[0]);
    if (dst.getIdx() == in_vec_idxs[0]) {
        TReg aux(aux_vec_idxs[0]);
        TReg src0(in_vec_idxs[0]);
        h->mov(aux.b16, src0.b16);
        mul0 = aux;
    }

    TReg mul1(in_vec_idxs[1]);
    if (dst.getIdx() == in_vec_idxs[1]) {
        TReg aux(aux_vec_idxs[0]);
        TReg src1(in_vec_idxs[1]);
        h->mov(aux.b16, src1.b16);
        mul1 = aux;
    }

    if (dst.getIdx() != in_vec_idxs[2]) {
        TReg src2(in_vec_idxs[2]);
        h->mov(dst.b16, src2.b16);
    }

    switch (exec_prc_) {
        case element::f16: {
            h->fmla(dst.h, mul0.h, mul1.h);
            break;
        }
        case element::f32: {
            h->fmla(dst.s, mul0.s, mul1.s);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16, element::f16}, {element::f32, element::f32, element::f32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_count() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if ((exec_prc_ != ov::element::f16) && (exec_prc_ != ov::element::f32)) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    switch (exec_prc_) {
        case ov::element::f16: {
            h->uni_fmul(dst.h, src0.h, src1.h);
            break;
        }
        case ov::element::f32: {
            h->uni_fmul(dst.s, src0.s, src1.s);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// POWER ///
jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const std::shared_ptr<ov::Node>& node,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, node, exec_prc) {
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        OPENVINO_THROW("Can't cast to snippets::op::PowerStatic");
    }

    power = powerStaticNode->get_power();
    scale = 1.f;
    shift = 0.f;

    prepare_table();
}

jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const float power,
                                                   const float scale,
                                                   const float shift,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, exec_prc),
                                                     power(power),
                                                     scale(scale),
                                                     shift(shift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_gprs_count() const { return 2; }

void jit_power_static_emitter::register_table_entries() {
    push_arg_entry_of("power", dnnl::impl::float2int(power), true);
    switch (exec_prc_) {
        case ov::element::f16: {
            push_arg_entry_of("scale", dnnl::impl::utils::bit_cast<int16_t>(static_cast<float16>(scale)), true);
            push_arg_entry_of("shift", dnnl::impl::utils::bit_cast<int16_t>(static_cast<float16>(shift)), true);
            break;
        }
        case ov::element::f32: {
            push_arg_entry_of("scale", dnnl::impl::float2int(scale), true);
            push_arg_entry_of("shift", dnnl::impl::float2int(shift), true);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

void jit_power_static_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if ((exec_prc_ != ov::element::f16) && (exec_prc_ != ov::element::f32)) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg dst = TReg(out_vec_idxs[0]);

    if (power == 0.f) {
        switch (exec_prc_) {
            case ov::element::f16: {
                h->fmov(dst.h, 1.);
                break;
            }
            case ov::element::f32: {
                h->fmov(dst.s, 1.);
                break;
            }
            default: {
                OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
            }
        }
        return;
    }

    bool get_from_dst = false;
    const auto src = [&in_vec_idxs, &out_vec_idxs, &get_from_dst]() -> TReg {
        return get_from_dst ? TReg(out_vec_idxs[0]) : TReg(in_vec_idxs[0]);
    };

    TReg aux = TReg(aux_vec_idxs[0]);
    if (scale != 1.f) {
        auto adr = table_val2("scale");
        switch (exec_prc_) {
            case ov::element::f16: {
                h->ld1r(aux.s, adr);
                h->ld1r(aux.h, adr);
                h->fmul(dst.h, src().h, aux.h);
                break;
            }
            case ov::element::f32: {
                h->ld1r(aux.s, adr);
                h->fmul(dst.s, src().s, aux.s);
                break;
            }
            default: {
                OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
            }
        }
        get_from_dst = true;
    }

    if (shift != 0.f) {
        auto adr = table_val2("shift");
        switch (exec_prc_) {
            case ov::element::f16: {
                h->ld1r(aux.h, adr);
                h->fadd(dst.h, src().h, aux.h);
                break;
            }
            case ov::element::f32: {
                h->ld1r(aux.s, adr);
                h->fadd(dst.s, src().s, aux.s);
                break;
            }
            default: {
                OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
            }
        }
        get_from_dst = true;
    }

    if (power == 1.f) {
        if (!get_from_dst && (in_vec_idxs[0] != dst.getIdx())) {
            h->mov(dst.b16, src().b16);
        }
        return;
    }

    if (std::floor(power) == power && power > 0) {
        h->mov(aux.b16, src().b16);
        switch (exec_prc_) {
            case ov::element::f16: {
                h->fmov(dst.h, 1.);
                break;
            }
            case ov::element::f32: {
                h->fmov(dst.s, 1.);
                break;
            }
            default: {
                OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
            }
        }

        auto current_power = static_cast<size_t>(power);
        while (current_power > 0) {
            if (current_power & 1) {
                switch (exec_prc_) {
                    case ov::element::f16: {
                        h->fmul(dst.h, dst.h, aux.h);
                        break;
                    }
                    case ov::element::f32: {
                        h->fmul(dst.s, dst.s, aux.s);
                        break;
                    }
                    default: {
                        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
                    }
                }
            }
            if (current_power > 1) {
                switch (exec_prc_) {
                    case ov::element::f16: {
                        h->fmul(aux.h, aux.h, aux.h);
                        break;
                    }
                    case ov::element::f32: {
                        h->fmul(dst.s, dst.s, aux.s);
                        break;
                    }
                    default: {
                        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
                    }
                }
            }
            current_power = current_power >> 1;
        }
    } else {
        auto pow_f32_addr = reinterpret_cast<uintptr_t>(::powf);

        Xbyak_aarch64::XReg func_reg(aux_gpr_idxs[0]);
        h->mov(func_reg, pow_f32_addr);

        Xbyak_aarch64::SReg s0(0);
        Xbyak_aarch64::SReg s1(1);

        const std::unordered_set<size_t> exclude = {src().getIdx(), dst.getIdx()};
        store_context(exclude);
        const auto amount = exec_prc_ == ov::element::f32 ? 4 : 8;
        for (auto i = 0; i < amount; i++) {
            switch (exec_prc_) {
                case ov::element::f16: {
                    Xbyak_aarch64::HReg h0(0);
                    h->mov(h0, src().h[i]);
                    h->fcvt(s0, h0);
                    break;
                }
                case ov::element::f32: {
                    h->mov(s0, src().s[i]);
                    break;
                }
                default: {
                    OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
                }
            }
            h->ldr(s1, table_val("power"));

            h->str(Xbyak_aarch64::QReg(dst.getIdx()), pre_ptr(h->sp, -16));
            h->str(Xbyak_aarch64::QReg(src().getIdx()), pre_ptr(h->sp, -16));
            h->blr(func_reg);
            h->ldr(Xbyak_aarch64::QReg(src().getIdx()), post_ptr(h->sp, 16));
            h->ldr(Xbyak_aarch64::QReg(dst.getIdx()), post_ptr(h->sp, 16));

            switch (exec_prc_) {
                case ov::element::f16: {
                    Xbyak_aarch64::HReg h0(0);
                    h->fcvt(h0, s0);
                    h->mov(dst.h[i], TReg(0).h[0]);
                    break;
                }
                case ov::element::f32: {
                    h->mov(dst.s[i], TReg(0).s[0]);
                    break;
                }
                default: {
                    OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
                }
            }
        }
        restore_context(exclude);
    }
}

/// PRELU ///
jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_prelu_emitter::get_inputs_count() const { return 2; }

size_t jit_prelu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fcmge(dst.s, src1.s, 0.0);
    h->fmul(tmp.s, src1.s, src2.s);
    h->bsl(dst.b16, src1.b16, tmp.b16);
}

/// RELU ///
jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_relu_emitter::get_inputs_count() const { return 1; }

size_t jit_relu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if ((exec_prc_ != ov::element::f16) && (exec_prc_ != ov::element::f32)) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    switch (exec_prc_) {
        case ov::element::f16: {
            h->movi(tmp.h, 0);
            h->fmaxnm(dst.h, src.h, tmp.h);
            break;
        }
        case ov::element::f32: {
            h->movi(tmp.s, 0);
            h->fmaxnm(dst.s, src.s, tmp.s);
            break;
        }
        default: {
            OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
        }
    }
}

/// SUBTRACT ///
jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_subtract_emitter::get_inputs_count() const { return 2; }

void jit_subtract_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fsub(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
