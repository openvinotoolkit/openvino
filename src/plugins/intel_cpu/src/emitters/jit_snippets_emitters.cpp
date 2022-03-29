// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

KernelEmitter::KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
    if (!kernel)
        IE_THROW() << "KernelEmitter invoked with invalid op argument";
    body = kernel->region;
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out,
                              const std::vector<size_t> &pool,
                              const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool,
                                       const std::vector<size_t> &gpr) const {
//        if (in.size() != 2)
//            IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 2, got " << in.size();
        if (in.size() != 0)
            IE_THROW() << "KernelEmitter doesn't accept arguments.";
        if (out.size() != 0)
            IE_THROW() << "KernelEmitter got unexpected output arguments.";
//        const size_t num_params = in[0] + in[1];
//        if (num_params > SNIPPETS_MAX_SNIPPETS_DIMS)
//            IE_THROW() << "KernelEmitter supports only up to " << SNIPPETS_MAX_SNIPPETS_DIMS <<
//                       " parameters, got " << num_params;
//        const int64_t harness_num_dims = jcp.output_dims.size() - 1;
//        if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS)
//            IE_THROW() << "KernelEmitter supports harness with up to " << SNIPPETS_MAX_HARNESS_DIMS <<
//                       " dims, got " << harness_num_dims;
}

void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& pool,
                              const std::vector<size_t>& gpr,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    h->preamble();
    for (auto& c : body) {
        c.first->emit_code(c.second.first, c.second.second, pool, gpr);
    }
        h->postamble();
}

TileSchedulerEmitter::TileSchedulerEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto tile_scheduler = ov::as_type_ptr<ngraph::snippets::op::TileScheduler>(n);
    if (!tile_scheduler)
        IE_THROW() << "TileSchedulerEmitter invoked with invalid op argument";
    if (!tile_scheduler->compile_params)
        IE_THROW() << "TileEmitter invoked without compile_params";
    vector_tile = tile_scheduler->vector_region;
    scalar_tile = tile_scheduler->scalar_region;
    vector_tile_body = std::dynamic_pointer_cast<TileEmitter>(vector_tile.first)->get_body();
    scalar_tile_body = std::dynamic_pointer_cast<TileEmitter>(vector_tile.first)->get_body();
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(tile_scheduler->compile_params);
}
void TileSchedulerEmitter::emit_code(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
//        todo: Enable validate arguments
//        validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileSchedulerEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& pool,
                                     const std::vector<size_t>& gpr,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];
    const size_t vector_size = in[2];
    const size_t num_params = num_inputs + num_outputs;
    const size_t outer_work_amount = jcp.scheduler_dims[0];
    const size_t inner_work_amount = jcp.scheduler_dims[1];
    const int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    const int64_t harness_num_dims = jcp.output_dims.size() - 1;

    // These are kernel runtime input arguments
    Reg64 reg_indexes   { dnnl::impl::cpu::x64::abi_param1 };
    Reg64 reg_const_params { dnnl::impl::cpu::x64::abi_param2 };

    Xbyak::Reg64 reg_tmp_64 { dnnl::impl::cpu::x64::abi_not_param1};
    Reg64 reg_amount = Reg64(reg64_tmp_start + num_params); // amount
    Label for_body;

    // If R15 is not used, reserve it for use in scalar to avoid redundant push-pop's.
    // todo: Do we need explicitly check that code contains ScalarEmitter?
    std::vector<size_t> local_gpr = reg64_tmp_start + num_params < 15 ? std::vector<size_t>{15} : std::vector<size_t>{};
    std::vector<Reg64> regs(num_params);

    auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t *offsets) {
        for (int j = 0; j < harness_num_dims; j++) {
            if (jcp.output_dims[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp_64, offsets[j]);
                h->imul(reg_tmp_64, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp_64);
            }
        }
    };
    for (auto i = 0; i < num_params; i++) {
        regs[i] = Reg64(reg64_tmp_start + i);
        if (i < num_inputs)
            h->mov(regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptrs_with_offsets(regs[i], &jcp.data_offsets[i * harness_num_dims]);
    }

    for (auto i = 0; i < num_params; i++)
        regs[i] = Reg64(reg64_tmp_start + i);

    auto emit_tiles = [&]() {
        auto process_tile =
                [&](bool body_condition, const std::vector<EmitterCode>& body,
                                    bool tile_condition, const EmitterCode& tile) {
            if (body_condition) {
                // emit Tile body directly if only one tile iteration is needed
                for (auto& c : body)
                    c.first->emit_code(c.second.first, c.second.second, pool, local_gpr);
            } else if (tile_condition) {
                // Need to set proper work amount for inner tiles before code emission
                h->mov(reg_amount, inner_work_amount);
                const ngraph::snippets::RegInfo &regInfo = tile.second;
                tile.first->emit_code(regInfo.first, regInfo.second, pool, local_gpr);
            }
        };
        process_tile(inner_work_amount == vector_size, vector_tile_body, inner_work_amount > vector_size, vector_tile);
        process_tile(inner_work_amount % vector_size == 1, scalar_tile_body, inner_work_amount % vector_size > 1, scalar_tile);
    };

    if (outer_work_amount == 1) {
        // emit code directly without looping over external dim
        emit_tiles();
    } else if (outer_work_amount > 1) {
        // We need to create a Loop in this case
        h->mov(reg_amount, outer_work_amount);
        h->L(for_body);
        {
            h->push(reg_amount);
            emit_tiles();
            h->pop(reg_amount);

            // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
            //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
            //   To overcome this limitation, we add appropriate negative offsets if necessary.
            for (auto i = 0; i < num_params; i++) {
                if (jcp.scheduler_offsets[i] != 0) {
                    h->add(regs[i], jcp.scheduler_offsets[i]);
                }
            }
            // Note that outer dimensions are always incremented by 1 (outer tiles are always scalar)
            h->sub(reg_amount, 1);
            h->cmp(reg_amount, 1);
            h->jge(for_body, CodeGenerator::T_NEAR);
        }
    }
}

TileEmitter::TileEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto tile = ov::as_type_ptr<ngraph::snippets::op::Tile>(n);
    if (!tile)
        IE_THROW() << "TileEmitter invoked with invalid op argument";
    body = tile->region;
}

void TileEmitter::emit_code(const std::vector<size_t> &in,
                            const std::vector<size_t> &out,
                            const std::vector<size_t> &pool,
                            const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileEmitter::validate_arguments(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    if (in.size() != 2)
        IE_THROW() << "TileEmitter got invalid number of inputs. Expected 4, got " << in.size();
    if (out.size() != 0)
        IE_THROW() << "TileEmitter got unexpected output arguments.";
    const size_t num_params = in[1];
    if (num_params > SNIPPETS_MAX_SNIPPETS_DIMS)
        IE_THROW() << "TileEmitter supports only up to " << SNIPPETS_MAX_SNIPPETS_DIMS <<
                   " parameters, got " << num_params;
}

void TileEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t inc = in[0];
    const size_t num_params = in[1];
    const int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    Reg64 amount = Reg64(reg64_tmp_start + num_params); // amount
    std::array<Label, 2> for_body;

    // If R15 is not used, reserve it for use in scalar to avoid redundant push-pop's.
    // todo: Do we need explicitly check that code contains ScalarEmitter?
    std::vector<size_t> local_gpr = reg64_tmp_start + num_params < 15 ? std::vector<size_t>{15} : std::vector<size_t>{};

    // Note that:
    // * Work amount must be set by TileScheduler that executes Tiles
    // * TileScheduler execute Tile only if it has to perform >= 1 iterations
    // Todo: remove this after tests
//        h->cmp(amount, inc);
//        h->jl(for_body[0], CodeGenerator::T_NEAR);
    h->L(for_body[1]);
    {
        h->push(amount);
        for (auto& c : body) {
            c.first->emit_code(c.second.first, c.second.second, pool, local_gpr);
        }
        h->pop(amount);
        h->sub(amount, inc);
        h->cmp(amount, inc);
        h->jge(for_body[1], CodeGenerator::T_NEAR);
    }
//        h->L(for_body[0]);
}

FakeBroadcastEmitter::FakeBroadcastEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    if (n->get_input_shape(0).empty())
        use_broadcast = true;
    else if (*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin())
        use_broadcast = true;
    else
        use_broadcast = false;
}

void FakeBroadcastEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out,
          const std::vector<size_t>& pool,
          const std::vector<size_t>& gpr,
          const ov::intel_cpu::emitter_context *emit_context) const {
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
void FakeBroadcastEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_src0 = Vmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    if (use_broadcast) {
        h->uni_vbroadcastss(vmm_dst, Xmm(in[0]));
    } else {
        h->uni_vmovups(vmm_dst, vmm_src0);
    }
}

ScalarEmitter::ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    value = dnnl::impl::cpu::x64::float2int(ov::as_type_ptr<ngraph::snippets::op::Scalar>(n)->cast_vector<float>()[0]);
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& pool,
                              const std::vector<size_t>& gpr,
                              const ov::intel_cpu::emitter_context *emit_context) const {
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
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_dst  = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}


MemoryEmitter::MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n), ea(getEA(n)) {
}

size_t MemoryEmitter::getEA(const std::shared_ptr<ov::Node>& n) {
    auto& rt = n->get_rt_info();
    size_t ea = 0;
    auto it = rt.find("effectiveAddress");
    if (it != rt.end()) {
        ea = it->second.as<int64_t>();
    } else {
        throw ov::Exception("effective address for Load generation cannot be determined");
    }
    return ea;
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out,
                             const std::vector<size_t>& pool,
                             const std::vector<size_t>& gpr,
                             const ov::intel_cpu::emitter_context *emit_context) const {
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
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 out_reg(ea);
    Vmm vmm_src0 = Vmm(in[0]);
    h->uni_vmovups(h->ptr[out_reg], vmm_src0);
    h->add(out_reg, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
}

ScalarStoreEmitter::ScalarStoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                       const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
}

void ScalarStoreEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out,
                                   const std::vector<size_t>& pool,
                                   const std::vector<size_t>& gpr,
                                   const ov::intel_cpu::emitter_context *emit_context) const {
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
void ScalarStoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 out_reg(ea);
    Xmm vmm_src0 = Xmm(in[0]);
    h->uni_vmovss(h->ptr[out_reg], vmm_src0);
    h->add(out_reg, sizeof(float));
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n)
                         : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
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
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(ea);
    Vmm vmm_src0 = Vmm(out[0]);
    h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

    if (shouldPostIncrement) {
        h->add(in_reg, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    }
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
}

void BroadcastLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& pool,
                                     const std::vector<size_t>& gpr,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
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
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(ea);
    Vmm vmm_src0 = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
}


ScalarLoadEmitter::ScalarLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                     const std::shared_ptr<ov::Node>& n)
                                    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
}

void ScalarLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                  const std::vector<size_t>& out,
                                  const std::vector<size_t>& pool,
                                  const std::vector<size_t>& gpr,
                                  const ov::intel_cpu::emitter_context *emit_context) const {
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
void ScalarLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(ea);
    Xmm vmm_src0 = Xmm(out[0]);
    h->uni_vmovss(vmm_src0, h->ptr[in_reg]);

    // Doesn't work if the same pointer comes with multiple load operations
    if (shouldPostIncrement) {
        h->add(in_reg, sizeof(float));
    }
}
}   // namespace intel_cpu
}   // namespace ov
