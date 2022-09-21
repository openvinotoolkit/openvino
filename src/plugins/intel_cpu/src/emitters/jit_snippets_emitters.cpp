// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
}

jit_container_emitter::jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                      const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(const std::vector<size_t> &vec_pool,  const std::vector<size_t> &gpr_pool,
                                                    std::set<size_t>& vecs_used, std::set<size_t>& gprs_used) {
    if (body.empty())
        IE_THROW() << "Cannot map registers for jit_container_emitter when its body is empty";
    auto abstract_to_physical = [](const std::vector<size_t>& abstract_regs, const std::vector<size_t>& regs_pool) {
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++)
            physical_regs[i] = regs_pool.at(abstract_regs[i]);
        return physical_regs;
    };
    for (auto& code : body) {
        const auto& emitter = code.first;
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = code.second;
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                // Note that gpr_to_gpr is used for high-level utility operations like Kernel/TileScheduler/Tile.
                // Input registers are not mapped in this case, since they contain utility info
                // (num_params, tile increment, etc.), but not reg indexes.
                in_physical_regs = std::move(in_abstract_regs);
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, gpr_pool));
                gprs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, gpr_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec_pool));
                gprs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                vecs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, gpr_pool));
                vecs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                gprs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, vec_pool));
                out_physical_regs = std::move(abstract_to_physical(out_abstract_regs, vec_pool));
                vecs_used.insert(in_physical_regs.begin(), in_physical_regs.end());
                vecs_used.insert(out_physical_regs.begin(), out_physical_regs.end());
                break;
            default:
                IE_THROW() << "Unhandled in_out type";
        }
        code.second = std::make_pair(in_physical_regs, out_physical_regs);
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(code.first))
            container->map_abstract_registers(vec_pool, gpr_pool, vecs_used, gprs_used);
    }
}

KernelEmitter::KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
    if (!kernel)
        IE_THROW() << "KernelEmitter invoked with invalid op argument";
    if (kernel->region.empty())
        IE_THROW() << "KernelEmitter invoked with empty body";
    body = kernel->region;
    if (!kernel->compile_params)
        IE_THROW() << "KernelEmitter invoked without compile_params";
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    std::iota(gp_regs_pool.begin(), gp_regs_pool.end(), 0);
    std::iota(vec_regs_pool.begin(), vec_regs_pool.end(), 0);
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         static_cast<size_t>(abi_param1.getIdx()),
                                         static_cast<size_t>(abi_param2.getIdx())});
    std::set<size_t> vecs_used, gprs_used;
    map_abstract_registers(vec_regs_pool, gp_regs_pool, vecs_used, gprs_used);
    remove_regs_from_pool(gp_regs_pool, gprs_used);
    remove_regs_from_pool(vec_regs_pool, vecs_used);
    // Remember used gprs to pass it to the TileSchedulerEmitter, so it can init them with appropriate data ptrs
    gp_regs_used = std::vector<size_t>(gprs_used.begin(), gprs_used.end());
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
    if (in.size() != 2)
        IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 2, got " << in.size();
    if (!out.empty())
        IE_THROW() << "KKernelEmitter got invalid number of outputs. Expected 0, got " << out.size();
}

void KernelEmitter::init_data_pointers(size_t num_inputs, size_t num_params,
                                              const Reg64& reg_indexes, const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    const int64_t harness_num_dims = jcp.output_dims.size() - 1;
    auto init_ptrs_with_offsets = [&](Reg64 pointer, const int64_t *offsets, Reg64 reg_tmp) {
        for (int j = 0; j < harness_num_dims; j++) {
            if (jcp.output_dims[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    for (auto i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        // we can use the last data_ptr_reg as tmp_reg until the last iteration, and reg_const_params then
        Reg64 reg_tmp = i < num_params-1 ? data_ptr_regs.back() : reg_const_params;
        init_ptrs_with_offsets(data_ptr_regs[i], &jcp.data_offsets[i * harness_num_dims], reg_tmp);
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& allocated_vec_regs,
                              const std::vector<size_t>& allocated_gp_regs,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    h->preamble();

    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];

    Reg64 reg_indexes = Reg64(abi_param1.getIdx());
    Reg64 reg_const_params = Reg64(abi_param2.getIdx());
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(gp_regs_used, data_ptr_regs);

    init_data_pointers(num_inputs, num_inputs + num_outputs, reg_indexes, reg_const_params, data_ptr_regs);
    // todo: emit_impl is a const method, so we can't just push_back unused regs to the gp_regs_pool.
    //  we need a more elegant approach to avoid a full copy here
    auto local_gpr_pool = gp_regs_pool;
    local_gpr_pool.push_back(static_cast<size_t>(reg_indexes.getIdx()));
    local_gpr_pool.push_back(static_cast<size_t>(reg_const_params.getIdx()));
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        if (auto tile_scheduler = std::dynamic_pointer_cast<TileSchedulerEmitter>(emitter))
            out_regs = gp_regs_used;
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, local_gpr_pool);
    }
    h->postamble();
}

TileSchedulerEmitter::TileSchedulerEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto tile_scheduler = ov::as_type_ptr<ngraph::snippets::op::TileScheduler>(n);
    if (!tile_scheduler)
        IE_THROW() << "TileSchedulerEmitter invoked with invalid op argument";
    if (!tile_scheduler->compile_params)
        IE_THROW() << "TileEmitter invoked without compile_params";
    body = {tile_scheduler->vector_region, tile_scheduler->scalar_region};
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
    if (in.size() != 3)
        IE_THROW() << "TileSchedulerEmitter got invalid number of inputs. Expected 3, got " << in.size();
    if (out.size() != in[0] + in[1])
        IE_THROW() << "TileSchedulerEmitter got invalid number of outputs. Expected " << in[0] + in[1] << " , got " << out.size();
    if (body.size() != 2)
        IE_THROW() << "TileSchedulerEmitter got invalid body size, expected 2 (vector & scalar TileEmitter), got " << body.size();
    if (!(std::dynamic_pointer_cast<TileEmitter>(body[0].first) && std::dynamic_pointer_cast<TileEmitter>(body[1].first)))
        IE_THROW() << "TileSchedulerEmitter can contain only TileEmitters inside its body";
}

void TileSchedulerEmitter::emit_tiles(const Reg64& reg_inner_amount, const std::vector<Reg64>& data_ptr_regs, size_t vector_size,
                                      const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    // TileAllocatedEmitter is just an alias to perform dynamic_pointer_cast only once and reuse it below several times
    using TileAllocatedEmitter = std::pair<std::shared_ptr<TileEmitter>, const ngraph::snippets::RegInfo&>;
    TileAllocatedEmitter vector_tile {std::dynamic_pointer_cast<TileEmitter>(body[0].first), body[0].second};
    TileAllocatedEmitter scalar_tile {std::dynamic_pointer_cast<TileEmitter>(body[1].first), body[1].second};
    const size_t inner_work_amount = jcp.scheduler_dims[1];
    auto process_tile =
        [&](const bool evaluate_once, const TileAllocatedEmitter& tile) {
            // If Tile is evaluated only once, then we can emit its body directly and skip work_amount decrements and checks
            if (evaluate_once) {
                tile.first->emit_body(vec_pool, gpr_pool);
            } else {
                std::vector<size_t> in_regs, out_regs;
                std::tie(in_regs, out_regs) = tile.second;
                // pass work_amount reg to Tile
                in_regs.push_back(static_cast<size_t>(reg_inner_amount.getIdx()));
                for (const auto& reg : data_ptr_regs)
                    out_regs.emplace_back(reg.getIdx());
                tile.first->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
            }
        };
    // todo: these optimizations should be performed on using Tile graph representation in the future
    bool vector_evaluate_once = false;
    if (inner_work_amount >= vector_size) {
        vector_evaluate_once = inner_work_amount < 2 * vector_size;
        // Need to set proper work amount for inner tiles if evaluated multiple times
        if (!vector_evaluate_once)
            h->mov(reg_inner_amount, inner_work_amount);
        process_tile(vector_evaluate_once, vector_tile);
    }
    if (inner_work_amount % vector_size >= 1) {
        bool scalar_evaluate_once = inner_work_amount % vector_size < 2;
        if (!scalar_evaluate_once) {
            // vector_tile is not executed, work_amount is not set
            if (inner_work_amount < vector_size) {
                h->mov(reg_inner_amount, inner_work_amount);
                // vector_tile is executed, but work_amount is neither set nor decremented appropriately.
            } else if (vector_evaluate_once) {
                vector_tile.first->emit_ptr_increments(data_ptr_regs);
                h->mov(reg_inner_amount, inner_work_amount - vector_size);
            }
            // else: vector_tile is executed multiple times, so work_amount is already set
        } else {
            if (vector_evaluate_once) {
                vector_tile.first->emit_ptr_increments(data_ptr_regs);
            }
        }
        process_tile(scalar_evaluate_once, scalar_tile);
    }
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
    const auto& data_ptr_reg_idxs(out);
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    // todo: emit_impl has const input args, so we can't just pop_back necessary regs from gpr_pool.
    //  we need a more elegant approach to avoid a full copy here. Similar problem is demonstrated in KernelEmitter
    auto local_gpr_pool = gpr_pool;
    Reg64 reg_outer_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    Reg64 reg_inner_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    Label for_body;
    const size_t outer_work_amount = jcp.scheduler_dims[0];
    if (outer_work_amount == 1) {
        // emit code directly without looping over external dim
        emit_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);
    } else if (outer_work_amount > 1) {
        // We need to create a Loop in this case
        h->mov(reg_outer_amount, outer_work_amount);
        h->L(for_body);
        {
            emit_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);

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

std::vector<AllocatedEmitter>& TileEmitter::get_nested_code() {
    return body;
}

TileEmitter::TileEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto tile = ov::as_type_ptr<ngraph::snippets::op::Tile>(n);
    if (!tile)
        IE_THROW() << "TileEmitter invoked with invalid op argument";
    body = tile->region;
    if (body.empty())
        IE_THROW() << "TileEmitter is invoked with empty body";
    num_inputs = tile->num_inputs;
    num_outputs = tile->num_outputs;
    io_dims = tile->io_dims;
    io_data_size = tile->io_data_size;
    increment = tile->increment;
    if (io_dims.size() != num_inputs + num_outputs)
        IE_THROW() << "TileEmitter constructor got inconsistent arguments. Check num_inputs + num_outputs == io_dims.size()";
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
    if (in.size() != 1)
        IE_THROW() << "TileEmitter got invalid number of inputs. Expected 1, got " << in.size();
    if (out.size() != io_dims.size())
        IE_THROW() << "TileEmitter got invalid number of outputs. Expected " << io_dims.size() << " , got " << out.size();
}

void TileEmitter::emit_body(const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    for (auto& code : body)
        code.first->emit_code(code.second.first, code.second.second, vec_pool, gpr_pool);
}

void TileEmitter::emit_ptr_increments(const std::vector<Reg64>& data_ptr_regs) const {
    for (size_t i = 0; i < num_inputs + num_outputs; i++) {
        // those with dims == 1 will be broadcasted, hence don't require increment
        if (io_dims[i] != 1)
            h->add(data_ptr_regs[i], increment * io_data_size[i]);
    }
}

void TileEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& vec_pool,
                            const std::vector<size_t>& gpr_pool,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    Reg64 work_amount = Reg64(static_cast<int>(in[0]));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(out, data_ptr_regs);
    Label for_body;
    // Note that:
    // * Work amount must be set by TileScheduler that executes Tiles
    // * TileScheduler executes Tile only if it has to perform >= 1 iterations
    h->L(for_body);
    emit_body(vec_pool, gpr_pool);
    emit_ptr_increments(data_ptr_regs);
    h->sub(work_amount, increment);
    h->cmp(work_amount, increment);
    h->jge(for_body, CodeGenerator::T_NEAR);
}

BroadcastMoveEmitter::BroadcastMoveEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    if (n->get_input_element_type(0) != n->get_output_element_type(0))
        IE_THROW() << "BroadcastMoveEmitter supports only equal input and output types but gets: "
            << n->get_input_element_type(0) << " and " << n->get_output_element_type(0);
    byte_size = n->get_input_element_type(0).size();
}

void BroadcastMoveEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out,
          const std::vector<size_t>& pool,
          const std::vector<size_t>& gpr,
          const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastMove emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastMoveEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Xmm xmm_src0 = Xmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    switch (byte_size) {
        case 4: h->uni_vbroadcastss(vmm_dst, xmm_src0); break;
        case 2: h->vpbroadcastw(vmm_dst, xmm_src0); break;
        case 1: h->vpbroadcastb(vmm_dst, xmm_src0); break;
        default: assert(!"unsupported data type");
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
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Scalar emitter doesn't support " << host_isa_;
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
    src_prc = InferenceEngine::details::convertPrecision(n->get_input_element_type(0));
    dst_prc = InferenceEngine::details::convertPrecision(n->get_output_element_type(0));
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "StoreEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    count = ov::as_type_ptr<ngraph::snippets::op::Store>(n)->get_count();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
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
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Store emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "LoadEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    count = ov::as_type_ptr<ngraph::snippets::op::Load>(n)->get_count();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
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
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Load emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
            IE_THROW() << "BroadcastEmitters support only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

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
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastLoad emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg]); break;
        default: assert(!"unsupported data type");
    }
}

LoadConvertEmitter::LoadConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    count = ov::as_type_ptr<ngraph::snippets::op::Load>(n)->get_count();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out,
                                   const std::vector<size_t>& pool,
                                   const std::vector<size_t>& gpr,
                                   const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "LoadConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadConvertEmitter::emit_data() const {
    load_emitter->emit_data();
}

StoreConvertEmitter::StoreConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    count = ov::as_type_ptr<ngraph::snippets::op::Store>(n)->get_count();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    }
}

void StoreConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out,
                                    const std::vector<size_t>& pool,
                                    const std::vector<size_t>& gpr,
                                    const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "StoreConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreConvertEmitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace intel_cpu
}   // namespace ov
