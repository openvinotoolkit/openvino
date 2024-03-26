// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_injectors.hpp"

#include <memory>
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

namespace utils {

class ValueTable {
public:
    ValueTable(
            dnnl::impl::cpu::aarch64::jit_generator *h,
            const std::multimap<std::string, jit_emitter::mapped_table_entry_t>* entry_map,
            const Xbyak_aarch64::XReg &p_table) : h(h), entry_map(entry_map), p_table(p_table) {
    }

    Xbyak_aarch64::AdrNoOfs value(const std::string &key, const size_t key_off_val_shift = 0) {
        const auto it = entry_map->find(key); // search an entry for a key
        assert(it != entry_map->end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? 16 : sizeof(jit_emitter::table_entry_val_t);
        const int32_t off = te.off + key_off_val_shift * scale;

        h->add_imm(h->X_DEFAULT_ADDR, p_table, off, h->X_TMP_0);
        return Xbyak_aarch64::ptr(h->X_DEFAULT_ADDR);
    }

private:
    dnnl::impl::cpu::aarch64::jit_generator* h;
    const std::multimap<std::string, jit_emitter::mapped_table_entry_t>* entry_map;
    const Xbyak_aarch64::XReg p_table;
};

class PushTable {
public:
    explicit PushTable(std::multimap<std::string, jit_emitter::mapped_table_entry_t>* entry_map) : entry_map(entry_map) {
    }

    void push(const std::string& key, const jit_emitter::table_entry_val_t val, const bool broadcast) {
        jit_emitter::mapped_table_entry_t te {0, val, broadcast};
        entry_map->insert(std::make_pair(key, te));
    }

private:
    std::multimap<std::string, jit_emitter::mapped_table_entry_t>* entry_map;
};
} // namespace utils

size_t jit_exp_injector::get_aux_vecs_count() { return 4; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_exp_injector::emit_impl(dnnl::impl::cpu::aarch64::jit_generator* h,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
                                 const ov::element::Type exec_prc,
                                 const std::vector<size_t> &in_vec_idxs,
                                 const std::vector<size_t> &aux_vec_idxs,
                                 const std::vector<size_t> &out_vec_idxs,
                                 const Xbyak_aarch64::XReg& p_table) {
    if (host_isa != dnnl::impl::cpu::aarch64::asimd) {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }

    if (exec_prc != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc.to_string());
    }

    utils::ValueTable table(h, &entry_map, p_table);

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);

    const TReg vmm_mask(aux_vec_idxs[3]);

    h->ld1r(vmm_aux0.s, table.value("exp_ln_flt_max_f"));
    h->fmin(vmm_dst.s, vmm_src.s, vmm_aux0.s);
    h->ld1r(vmm_aux0.s, table.value("exp_ln_flt_min_f"));
    h->fmax(vmm_dst.s, vmm_dst.s, vmm_aux0.s);

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->fcmgt(vmm_mask.s, vmm_src.s, vmm_aux0.s);
    h->mov(vmm_aux1.b16, vmm_dst.b16);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->ld1r(vmm_aux0.s, table.value("exp_log2ef"));
    h->ld1r(vmm_aux2.s, table.value("half"));
    h->fmla(vmm_aux2.s, vmm_dst.s, vmm_aux0.s);

    // tmp = floorf(fx)
    h->frintm(vmm_aux2.s, vmm_aux2.s);

    // keep vmm_src = fx for further computations
    h->mov(vmm_dst.b16, vmm_aux2.b16);

    // x = x - fx * ln2
    h->ld1r(vmm_aux0.s, table.value("ln2f"));
    h->fmls(vmm_aux1.s, vmm_aux2.s, vmm_aux0.s);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->ld1r(vmm_aux0.s, table.value("one"));
    h->fsub(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
    h->fcvtzs(vmm_aux2.s, vmm_dst.s);

    h->ld1r(vmm_aux0.s, table.value("exponent_bias"));
    h->add(vmm_aux2.s, vmm_aux2.s, vmm_aux0.s);

    h->sqshl(vmm_aux2.s, vmm_aux2.s, 23);

    // set zeroes at those points which were < log(FLT_MIN)
    h->and_(vmm_aux2.b16, vmm_mask.b16, vmm_aux2.b16);

    // compute polynomial
    h->ld1r(vmm_aux0.s, table.value("exp_pol5"));
    h->ld1r(vmm_dst.s, table.value("exp_pol4"));
    h->fmla(vmm_dst.s, vmm_aux1.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table.value("exp_pol3"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table.value("exp_pol2"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    h->ld1r(vmm_aux0.s, table.value("exp_pol1"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table.value("one"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    // y = y * 2^n
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux2.s);
    h->ld1r(vmm_aux0.s, table.value("two"));
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
}

void jit_exp_injector::push_entry_map(std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map) {
    utils::PushTable table(&entry_map);
    table.push("exp_ln_flt_max_f", 0x42b17218, true);
    table.push("exp_ln_flt_min_f", 0xc2aeac50, true);
    table.push("exp_log2ef", 0x3fb8aa3b, true);
    table.push("one", 0x3f800000, true);
    table.push("two", 0x40000000, true);
    table.push("half", 0x3f000000, true);
    table.push("ln2f", 0x3f317218, true);
    table.push("exponent_bias", 0x0000007f, true);
    table.push("exp_pol1", 0x3f7ffffb, true);
    table.push("exp_pol2", 0x3efffee3, true);
    table.push("exp_pol3", 0x3e2aad40, true);
    table.push("exp_pol4", 0x3d2b9d0d, true);
    table.push("exp_pol5", 0x3c07cfce, true);
}

size_t jit_sigmoid_injector::get_aux_vecs_count() {
    return jit_exp_injector::get_aux_vecs_count() + 2;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_sigmoid_injector::emit_impl(dnnl::impl::cpu::aarch64::jit_generator* h,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
                                     const ov::element::Type exec_prc,
                                     const std::vector<size_t> &in_vec_idxs,
                                     const std::vector<size_t> &aux_vec_idxs,
                                     const std::vector<size_t> &out_vec_idxs,
                                     const Xbyak_aarch64::XReg& p_table) {
    if (exec_prc != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc.to_string());
    }

    utils::ValueTable table(h, &entry_map, p_table);

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(jit_exp_injector::get_aux_vecs_count() + 1);

    const TReg vmm_mask(jit_exp_injector::get_aux_vecs_count());

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_mask for the mask as exp_compute does not use it.
    // we store the original sign and make x negative
    h->eor(vmm_aux0.b16, vmm_aux0.b16, vmm_aux0.b16);
    h->fcmgt(vmm_mask.s, vmm_src.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table.value("sign_mask"));
    h->orr(vmm_aux0.b16, vmm_src.b16, vmm_aux0.b16);

    jit_exp_injector::emit_impl<dnnl::impl::cpu::aarch64::asimd>(
            h,
            host_isa,
            entry_map,
            exec_prc,
            { vmm_aux0.getIdx() },
            aux_vec_idxs,
            out_vec_idxs,
            p_table);

    // dup exp(x)
    h->mov(vmm_aux1.b16, vmm_dst.b16);
    // (exp(x) + 1)
    h->ld1r(vmm_aux0.s, table.value("one"));
    h->fadd(vmm_aux1.s, vmm_aux1.s, vmm_aux0.s);
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_dst.s, vmm_dst.s, vmm_aux1.s);

    // Now we have to apply the "symmetry" based on original sign
    h->ld1r(vmm_aux2.s, table.value("one"));
    h->fsub(vmm_aux2.s, vmm_aux2.s, vmm_dst.s);

    h->bsl(vmm_mask.b16, vmm_aux2.b16, vmm_dst.b16);
    h->mov(vmm_dst.b16, vmm_mask.b16);
}

void jit_sigmoid_injector::push_entry_map(std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map) {
    jit_exp_injector::push_entry_map(entry_map);

    utils::PushTable table(&entry_map);
    table.push("sign_mask", 0x807fffff, true);
}

template void jit_exp_injector::emit_impl<dnnl::impl::cpu::aarch64::asimd>(
    dnnl::impl::cpu::aarch64::jit_generator* h,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
    const ov::element::Type exec_prc,
    const std::vector<size_t> &in_vec_idxs,
    const std::vector<size_t> &aux_vec_idxs,
    const std::vector<size_t> &out_vec_idxs,
    const Xbyak_aarch64::XReg& p_table);

template void jit_sigmoid_injector::emit_impl<dnnl::impl::cpu::aarch64::asimd>(
    dnnl::impl::cpu::aarch64::jit_generator* h,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const std::multimap<std::string, jit_emitter::mapped_table_entry_t>& entry_map,
    const ov::element::Type exec_prc,
    const std::vector<size_t> &in_vec_idxs,
    const std::vector<size_t> &aux_vec_idxs,
    const std::vector<size_t> &out_vec_idxs,
    const Xbyak_aarch64::XReg& p_table);

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
