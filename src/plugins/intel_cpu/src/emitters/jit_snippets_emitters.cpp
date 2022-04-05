// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {
// Define some helper functions in the anonymous namespace
namespace {

void remove_regs_from_pool(std::vector<size_t>& pool, const std::vector<size_t>& to_remove) {
    // It's important to keep the order of other elements
    for (const auto &reg_num : to_remove) {
        pool.erase(std::remove(pool.begin(), pool.end(), reg_num), pool.end());
    }
}
void add_regs_to_pool(std::vector<size_t>& pool, const std::vector<size_t>& to_add) {
    std::vector<size_t> sorted_to_add{to_add};
    std::sort(sorted_to_add.begin(), sorted_to_add.end());
    std::vector<size_t> result(pool.size() + to_add.size());

    std::merge(pool.begin(), pool.end(), sorted_to_add.begin(), sorted_to_add.end(), result.begin());
    pool = std::move(result);
}
/*
 * This function calls emitters for code and maps abstract registers (emitter args to physical ones)
 */
EmitterCode map_regs(const EmitterCode& code, const std::vector<size_t> &vec,  const std::vector<size_t> &gpr) {
    std::vector<size_t> gp_regs_pool{gpr};
    const auto& emitter = code.first;
    std::vector<size_t> in_abstract_regs;
    std::vector<size_t> out_abstract_regs;
    std::tie(in_abstract_regs, out_abstract_regs) = code.second;
    std::vector<size_t> in_physical_regs;
    std::vector<size_t> out_physical_regs;
    auto abstract_to_physical = [](const std::vector<size_t>& abstract_regs,
                                                          const std::vector<size_t>& regs_pool) {
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++)
            physical_regs[i] = regs_pool[abstract_regs[i]];
        return physical_regs;
    };
    switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
        case gpr_to_gpr:
//                // Not regs, but utility info, see the emitter for details
//                in_physical_regs = std::move(in_abstract_regs);
//                // out_abstract_regs are expected to be empty for now, but may be needed in future
//                out_physical_regs = std::move(out_abstract_regs);
//                if (std::dynamic_pointer_cast<TileSchedulerEmitter>(emitter) != nullptr) {
//                    out_physical_regs.push_back(dnnl::impl::cpu::x64::abi_param1.getIdx());
//                    out_physical_regs.push_back(dnnl::impl::cpu::x64::abi_param2.getIdx());
//                } else if (std::dynamic_pointer_cast<TileEmitter>(emitter) != nullptr) {
//                    // out_abstract_regs are expected to be empty for now, but may be needed in future
//                    out_physical_regs.push_back(dnnl::impl::cpu::x64::abi_param2.getIdx());
//                }
//                remove_regs_from_pool(gp_regs_pool, out_physical_regs);
            break;
        case gpr_to_vec:
            // Load Emmitters
            in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, gpr));
            out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec));
            break;
        case vec_to_gpr:
            in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec));
            out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, gpr));
            break;
        case vec_to_vec:
            in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec));
            out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec));
            break;
        default:
            IE_THROW() << "Unhandled in_out type";
    }
    return {emitter, {in_physical_regs, out_physical_regs}};
}
} // namespace

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
        if (in.size() != 1)
            IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 1, got " << in.size();
        if (out.size() != 0)
            IE_THROW() << "KernelEmitter got unexpected output arguments.";
        if (in[0] > SNIPPETS_MAX_SNIPPETS_DIMS)
            IE_THROW() << "KernelEmitter supports only up to " << SNIPPETS_MAX_SNIPPETS_DIMS <<
                       " parameters, got " << in[0];
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
    auto num_params = in[0];
    // Initialize pools of gp and vec registers
    std::vector<size_t> gp_regs_pool(16);
    std::iota(gp_regs_pool.begin(), gp_regs_pool.end(), 0);
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2 to process input arguments in TileScheduler
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         static_cast<size_t>(dnnl::impl::cpu::x64::abi_param1.getIdx()),
                                         static_cast<size_t>(dnnl::impl::cpu::x64::abi_param2.getIdx())});

    std::vector<size_t> data_ptr_regs(num_params);
    for (int i = 0; i < num_params; i++) {
        data_ptr_regs[i] = gp_regs_pool[i];
    }
    remove_regs_from_pool(gp_regs_pool, data_ptr_regs);

    std::vector<size_t> vec_regs_pool(16);
    std::iota(vec_regs_pool.begin(), vec_regs_pool.end(), 0);

    h->preamble();
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs;
        std::vector<size_t> out_regs;
        std::tie(in_regs, out_regs) = c.second;
        if (std::dynamic_pointer_cast<TileSchedulerEmitter>(emitter) != nullptr) {
            in_regs.push_back(static_cast<size_t>(dnnl::impl::cpu::x64::abi_param1.getIdx()));
            in_regs.push_back(static_cast<size_t>(dnnl::impl::cpu::x64::abi_param2.getIdx()));
            for (const auto reg_num : data_ptr_regs)
                out_regs.push_back(reg_num);
        }
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
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
    scalar_tile_body = std::dynamic_pointer_cast<TileEmitter>(scalar_tile.first)->get_body();
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(tile_scheduler->compile_params);
}
void TileSchedulerEmitter::emit_code(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}
void TileSchedulerEmitter::validate_arguments(const std::vector<size_t> &in,
                                     const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool,
                                     const std::vector<size_t> &gpr) const {
    if (in.size() != 5)
        IE_THROW() << "TileSchedulerEmitter got invalid number of inputs. Expected 5, got " << in.size();
    if (out.size() != in[0] + in[1])
        IE_THROW() << "TileSchedulerEmitter got invalid number of outputs. Expected " << in[0] + in[1] << " , got " << out.size();
}

void TileSchedulerEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& vec_pool,
                                     const std::vector<size_t>& gpr_pool,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];
    const size_t vector_size = in[2];
    const size_t num_params = num_inputs + num_outputs;
    const size_t outer_work_amount = jcp.scheduler_dims[0];
    const size_t inner_work_amount = jcp.scheduler_dims[1];
    const int64_t harness_num_dims = jcp.output_dims.size() - 1;

    // It is critical that reg_outer_amount and reg_inner_amount represent the
    // first two runtime arguments, since they are used to calculating offsets
    Reg64 reg_outer_amount = Reg64(static_cast<int>(in[3]));
    Reg64 reg_inner_amount = Reg64(static_cast<int>(in[4]));
    std::vector<size_t> gp_regs_pool(gpr_pool);

    Label for_body;

    Reg64 reg_indexes = reg_outer_amount;
    Reg64 reg_const_params = reg_inner_amount;
    auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t *offsets, Reg64 reg_tmp) {
        for (int j = 0; j < harness_num_dims; j++) {
            if (jcp.output_dims[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    std::vector<Reg64> data_ptr_regs(num_params);
    for (auto i = 0; i < num_params; i++) {
        data_ptr_regs[i] = Reg64(static_cast<int>(out[i]));
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        // we can use the last data_ptr_reg as tmp_reg until the last iteration, and reg_const_params then
        Reg64 reg_tmp = i < num_params-1 ? Reg64(static_cast<int>(out.back())) : reg_const_params;
        init_ptrs_with_offsets(data_ptr_regs[i], &jcp.data_offsets[i * harness_num_dims], reg_tmp);
    }
    auto emit_tiles = [&]() {
        bool inner_work_amount_is_set = false;
        auto process_tile =
                [&](bool body_condition, const std::vector<EmitterCode>& body,
                                    bool tile_condition, const EmitterCode& tile) {
            if (body_condition) {
                // emit Tile body directly if only one tile iteration is needed
                for (auto& original_code : body) {
                    auto code = map_regs(original_code, vec_pool, out);
                    code.first->emit_code(code.second.first, code.second.second, {}, gp_regs_pool);
                }
            } else if (tile_condition) {
                // Need to set proper work amount for inner tiles before code emission
                if (!inner_work_amount_is_set) {
                    h->mov(reg_inner_amount, inner_work_amount);
                    inner_work_amount_is_set = true;
                }
                std::vector<size_t> in_regs, out_regs;
                std::tie(in_regs, out_regs) = tile.second;
                // pass work_amount to Tile
                in_regs.push_back(static_cast<size_t>(dnnl::impl::cpu::x64::abi_param2.getIdx()));
                // append data_ptr regs, since Tile will need them to map
                for (const auto reg : data_ptr_regs)
                    out_regs.push_back(static_cast<size_t>(reg.getIdx()));
                tile.first->emit_code(in_regs, out_regs, vec_pool, gp_regs_pool);
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
        h->mov(reg_outer_amount, outer_work_amount);
        h->L(for_body);
        {
            emit_tiles();

            // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
            //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
            //   To overcome this limitation, we add appropriate negative offsets if necessary.
            for (auto i = 0; i < num_params; i++) {
                if (jcp.scheduler_offsets[i] != 0) {
                    h->add(data_ptr_regs[i], jcp.scheduler_offsets[i]);
                }
            }
            // Note that outer dimensions are always incremented by 1 (outer tiles are always scalar)
            h->sub(reg_outer_amount, 1);
            h->cmp(reg_outer_amount, 1);
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
    if (in.size() != 3)
        IE_THROW() << "TileEmitter got invalid number of inputs. Expected 2, got " << in.size();
    if (out.size() != in[0])
        IE_THROW() << "TileEmitter got invalid number of outputs. Expected " << in[0] << " , got " << out.size();
}

void TileEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& vec_pool,
                            const std::vector<size_t>& gpr_pool,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t inc = in[1];
    // Todo: Note that both Tiles use the same reg for amount and this is a problem,
    //  since we can't just pop it from the pool (we don't know whether it's first/second or the only tile at this point)
    Reg64 amount = Reg64(static_cast<int>(in[2]));
    Label for_body;

    // If R15 is not used, reserve it for use in scalar to avoid redundant push-pop's.
    // todo: Do we need explicitly check that code contains ScalarEmitter?
//    std::vector<size_t> local_gpr = {(size_t) dnnl::impl::cpu::x64::abi_not_param1.getIdx()};

    // Note that:
    // * Work amount must be set by TileScheduler that executes Tiles
    // * TileScheduler execute Tile only if it has to perform >= 1 iterations
    h->L(for_body);
    {
        for (auto& original_code : body) {
            auto code = map_regs(original_code, vec_pool, out);
            code.first->emit_code(code.second.first, code.second.second, {}, gpr_pool);
        }
        h->sub(amount, inc);
        h->cmp(amount, inc);
        h->jge(for_body, CodeGenerator::T_NEAR);
    }
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
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out,
                             const std::vector<size_t>& pool,
                             const std::vector<size_t>& gpr,
                             const ov::intel_cpu::emitter_context *emit_context) const {
    std::cerr << "Store: in :";
    for (auto o : in)
        std::cerr << o << " ";
    std::cerr << "\n";
    std::cerr << "Store: out : ";
    for (auto o : out)
        std::cerr << o << " ";
    std::cerr << "\n";
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
    Reg64 out_reg(static_cast<int>(out[0]));
    Vmm vmm_src0 = Vmm(in[0]);
    h->uni_vmovups(h->ptr[out_reg], vmm_src0);
    h->add(out_reg, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
}

ScalarStoreEmitter::ScalarStoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                       const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
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
    Reg64 out_reg(static_cast<int>(out[0]));
    Xmm vmm_src0 = Xmm(in[0]);
    h->uni_vmovss(h->ptr[out_reg], vmm_src0);
    h->add(out_reg, sizeof(float));
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n)
                         : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    std::cerr << "LoadEmitterInvoked\n";
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    std::cerr << "Load: in :";
    for (auto o : in)
        std::cerr << o << " ";
    std::cerr << "\n";
    std::cerr << "Load: out : ";
    for (auto o : out)
        std::cerr << o << " ";
    std::cerr << "\n";
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
    Reg64 in_reg(static_cast<int>(in[0]));
    Vmm vmm_src0 = Vmm(out[0]);
    h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

    if (shouldPostIncrement) {
        h->add(in_reg, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    }
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
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
    Reg64 in_reg(in[0]);
    Vmm vmm_src0 = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
}


ScalarLoadEmitter::ScalarLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                     const std::shared_ptr<ov::Node>& n)
                                    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
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
    Reg64 in_reg(static_cast<int>(in[0]));
    Xmm vmm_src0 = Xmm(out[0]);
    h->uni_vmovss(vmm_src0, h->ptr[in_reg]);

    // Doesn't work if the same pointer comes with multiple load operations
    if (shouldPostIncrement) {
        h->add(in_reg, sizeof(float));
    }
}
}   // namespace intel_cpu
}   // namespace ov
