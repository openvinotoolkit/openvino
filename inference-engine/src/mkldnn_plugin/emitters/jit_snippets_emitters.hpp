// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_emitter.hpp"

namespace MKLDNNPlugin {

class KernelEmitter : public jit_emitter {
public:
    KernelEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    // region
    const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        code = ngraph::as_type_ptr<ngraph::snippets::op::Kernel>(n)->region;
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
        auto tile_rank = in[0]; // count of tile dimensions
        auto nparams = in[1];
        int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        Xbyak::Reg64 param    { dnnl::impl::cpu::x64::abi_param1 }; // RDI
        Xbyak::Reg64 schedule { dnnl::impl::cpu::x64::abi_param3 }; // RDX

        h->preamble();

        std::vector<Xbyak::Reg64> regs(nparams);
        for (auto i = 0; i < nparams; i++) {
            regs[i] = Xbyak::Reg64(reg64_tmp_start + i);
            h->mov(regs[i], h->ptr[param + i * sizeof(int64_t)]);
        }

        // external amount
        Xbyak::Reg64 amount = Xbyak::Reg64(reg64_tmp_start + nparams);
        h->mov(amount, h->ptr[schedule + (tile_rank - 1) * sizeof(int64_t)]);

        for (auto& c : code) {
            c.first->emit_code(c.second.first, c.second.second);
        }

        h->postamble();
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};

class TileEmitter : public jit_emitter {
public:
    TileEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        code = ngraph::as_type_ptr<ngraph::snippets::op::Tile>(n)->region;
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
        const size_t inc = in[0];
        auto nparams = in[1];
        auto dim = in[2]; // number of tile dimension
        const int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        Xbyak::Reg64 amount = Xbyak::Reg64(reg64_tmp_start + nparams); // amount
        Xbyak::Reg64 offsets { dnnl::impl::cpu::x64::abi_param2 }; // RSI
        Xbyak::Reg64 shedule { dnnl::impl::cpu::x64::abi_param3 }; // RDX

        std::vector<Xbyak::Reg64> regs(nparams);
        if (dim > 0) {
            for (auto i = 0; i < nparams; i++) {
                regs[i] = Xbyak::Reg64(reg64_tmp_start + i);
            }
        }

        std::array<Xbyak::Label, 2> for_body;
        // loop_entry()
        if (dim == 0 && inc != 1)
            h->mov(amount, h->ptr[shedule + dim * sizeof(int64_t)]);

        h->cmp(amount, inc);
        h->jl(for_body[1], Xbyak::CodeGenerator::T_NEAR);

        // loop_body()
        h->L(for_body[0]); {
            h->push(amount);
            for (auto& c : code) {
                c.first->emit_code(c.second.first, c.second.second);
            }
            h->pop(amount);

            if (dim > 0) {
                for (auto i = 0; i < nparams; i++) {
                    h->add(regs[i], h->ptr[offsets + i * sizeof(int64_t)]);
                }
            }

            // loop_advance()
            h->sub(amount, inc);
            h->cmp(amount, inc);
            h->jge(for_body[0], Xbyak::CodeGenerator::T_NEAR);
        }

        h->L(for_body[1]);
    }

    // A = <42, 17>
    // B = < 1, 17>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //     auto b = *ptr1; ptr1 += vlan; // vector/scalar load
    //   }
    //   ptr0 -= 0*dom_1;
    //   ptr1 -= 1*dom_1;
    // }

    // broadcast by MVD is extra case
    // A = <42, 17>
    // B = <42,  1>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //     auto b = *ptr1;  // broadcast load
    //   }
    //   ptr0 -= 0*dom_1;
    //   ptr1 += sizeof(ptr1[0]); //ptr1 -= -sizeof(ptr1[0]);
    // }

    // A = <42, 17, 31>
    // B = < 1, 17, 31>
    // for (auto k = 0; k < dom_0; k++) { // 42
    //   for (auto n = 0; n < dom_1; n++) { // 17
    //     for (auto m = 0; m < dom_2; m++) { // 31
    //       auto a = *ptr0; ptr0 += vlan; // vector/scalar load
    //       auto b = *ptr1; ptr1 += vlan; // vector/scalar load
    //     }
    //   }
    //   ptr0 -= 0*dom_1*dom2;
    //   ptr1 -= 1*dom_1*dom2;
    // }
    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
    }
};

class FakeBroadcastEmitter : public jit_emitter {
public:
    FakeBroadcastEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), use_broadcast(*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin()) {
    }
    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Vmm vmm_src0 = Vmm(in[0]);
        Vmm vmm_dst  = Vmm(out[0]);

        if (use_broadcast) {
            h->uni_vbroadcastss(vmm_dst, Xbyak::Xmm(in[0]));
        } else {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }
    }

private:
    bool use_broadcast;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        auto out_shape = n->output(0).get_tensor().get_shape();
        if (out_shape == ngraph::Shape() || ngraph::shape_size(out_shape) == 1) {
            value = mkldnn::impl::cpu::x64::float2int(ngraph::as_type_ptr<ngraph::snippets::op::Scalar>(n)->cast_vector<float>()[0]);
        }

        push_arg_entry_of("scalar", value, true);
        prepare_table();
    }

    size_t get_inputs_num() const override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Vmm vmm_dst  = Vmm(out[0]);
        h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
    }

private:
    int32_t value;
};

///
/// Memory emitters:
///
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different tiles. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
    }

    size_t get_inputs_num() const override {return 1;}

protected:
    static auto getEA(const std::shared_ptr<ngraph::Node>& n) -> size_t {
        auto& rt = n->get_rt_info();
        size_t ea = 0;
        if (auto rinfo = rt["effectiveAddress"]) {
            ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
        } else {
            throw ngraph::ngraph_error("effective address for Load generation cannot be determined");
        }
        return ea;
    }

    size_t ea;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                    Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Xbyak::Reg64 out_reg(ea);
        Vmm vmm_src0 = Vmm(in[0]);
        h->uni_vmovups(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    }
};

class ScalarStoreEmitter : public MemoryEmitter {
public:
    ScalarStoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                        Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Xbyak::Reg64 out_reg(ea);
        Xbyak::Xmm vmm_src0 = Xbyak::Xmm(in[0]);
        h->movss(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, sizeof(float));
    }
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    // result of canonizing broadcast by C. Fixe it=)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Xbyak::Reg64 in_reg(ea);
        Vmm vmm_src0 = Vmm(out[0]);
        h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

        if (shouldPostIncrement) {
            h->add(in_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
        }
    }

private:
    bool shouldPostIncrement;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Xbyak::Reg64 in_reg(ea);
        Vmm vmm_src0 = Vmm(out[0]);

        // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
        // key point here is not to add post-increment, it might be fixed by some other approach in future
        h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
    }
};

class ScalarLoadEmitter : public MemoryEmitter {
public:
    ScalarLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
            emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
            emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
        } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
            emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
        } else {
            IE_THROW() << host_isa_;
            assert(!"unsupported isa");
        }
    }

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                            Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
        Xbyak::Reg64 in_reg(ea);
        Xbyak::Xmm vmm_src0 = Xbyak::Xmm(out[0]);
        h->movss(vmm_src0, h->ptr[in_reg]);

        // Doesn't work if the same pointer comes with multiple load operations
        if (shouldPostIncrement) {
            h->add(in_reg, sizeof(float));
        }
    }

private:
    bool shouldPostIncrement;
};

} // namespace MKLDNNPlugin