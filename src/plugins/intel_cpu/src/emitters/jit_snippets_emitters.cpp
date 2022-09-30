// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "jit_snippets_emitters.hpp"
#include "snippets/op/subgraph.hpp"

using namespace Xbyak;
using ngraph::snippets::op::Subgraph;

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
                // todo: Note that TileBeginEmitter and TileEndEmitter demonstrate new paradigm,
                //  where all utility emitters are align with conventional Op emitters
                if (std::dynamic_pointer_cast<TileBeginEmitter>(emitter) ||
                        std::dynamic_pointer_cast<TileEndEmitter>(emitter))
                    in_physical_regs = std::move(abstract_to_physical(in_abstract_regs, gpr_pool));
                else
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
    if (kernel->compile_params == nullptr)
        IE_THROW() << "KernelEmitter invoked with op::Kernel that contains no compile_params";
    body = kernel->region;
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
        IE_THROW() << "KernelEmitter got invalid number of outputs. Expected 0, got " << out.size();
    const auto num_params = in[0] + in[1];
    if (gp_regs_used.size() != num_params)
        IE_THROW() << "KernelEmitter arguments are inconsistent with the gpr_regs_used size: in[0] + in[1] = "
        << num_params << " gp_regs_used.size() = " << gp_regs_used.size();
}

void KernelEmitter::init_data_pointers(size_t num_inputs, size_t num_params,
                                              const Reg64& reg_indexes, const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    // master_shape size must be valid in both static and dynamic cases
    const int64_t offsetRank = jcp.master_shape.size() - 1;
    std::function<void(Reg64, size_t, Reg64)> init_ptr_with_offset;
    if (jcp.is_static) {
        init_ptr_with_offset = [&](Reg64 pointer, size_t offset_start_index, Reg64 reg_tmp) {
            const int64_t *offsets =  jcp.data_offsets + offset_start_index;
            for (int j = 0; j < offsetRank; j++) {
                if (jcp.master_shape[j] != 1 && offsets[j] != 0) {
                    h->mov(reg_tmp, offsets[j]);
                    h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                    h->add(pointer, reg_tmp);
                }
            }
        };
    } else {
        init_ptr_with_offset = [&](Reg64 pointer, size_t offset_displ, Reg64 reg_tmp) {
            const size_t data_offests_displ = GET_OFF(data_offsets) + offset_displ * sizeof(int64_t);
            // todo: we can pre-filter data_offsets so that only (master_shape[k] != 1 && offsets[k] != 0) are stored there
            //  but we'll need an additional index array to specify appropriate "k" values for every input
            //  * size_t num_non_zero_offsets[num_params] - specifies number of non-zero offsets for every input
            //  * size_t offsetted_indexes* - points to memory chunk sizeof(sum(num_non_zero_offsets) * sizeof(size_t)) -
            //                                  specifies indexes of input indexes (reg_index) that need an offset
            //  * size_t data_offsets* - the same size as offsetted_indexes - offset values for input indexes
            for (int j = 0; j < offsetRank; j++) {
                h->mov(reg_tmp, h->ptr[reg_const_params + data_offests_displ + j * sizeof(int64_t)]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        };
    }
    const bool last_iter_explicitly = gp_regs_pool.empty();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs.back() : Reg64(gp_regs_pool.back());
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], i * offsetRank, reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to TileScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_const_params;
        if (jcp.is_static) {
            // can corrupt reg_const_params, since we won't use it anymore
            init_ptr_with_offset(data_ptr_regs[i], i * offsetRank, reg_tmp);
        } else {
            // have to restore reg_tmp explicitly in dynamic case, can use stack or vector reg
            h->push(reg_tmp);
            init_ptr_with_offset(data_ptr_regs[i], i * offsetRank, reg_tmp);
            h->pop(reg_tmp);
        }
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& vec_pool,
                              const std::vector<size_t>& gpr_pool,
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
    // we won't need indexes in both static and dynamic cases, since offsets are already calculated
    local_gpr_pool.push_back(static_cast<size_t>(reg_indexes.getIdx()));
//    if (jcp.is_static) {
//        local_gpr_pool.push_back(static_cast<size_t>(reg_const_params.getIdx()));
//    }
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        if (auto tile_scheduler = std::dynamic_pointer_cast<TileSchedulerEmitter>(emitter)) {
            // dynamic TileScheduler needs const runtime params
            if (!jcp.is_static) {
                in_regs.push_back(static_cast<size_t>(reg_const_params.getIdx()));
            }
            out_regs = gp_regs_used;
        }
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, local_gpr_pool);
    }
    h->postamble();
}

TileSchedulerEmitter::TileSchedulerEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto tile_scheduler = ov::as_type_ptr<ngraph::snippets::op::TileScheduler>(n);
    if (!tile_scheduler)
        IE_THROW() << "TileSchedulerEmitter invoked with invalid op argument";
    if (tile_scheduler->compile_params == nullptr)
        IE_THROW() << "TileSchedulerEmitter invoked with op::TileScheduler that contains no compile_params";
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
    if (jcp.is_static && in.size() != 3)
        IE_THROW() << "TileSchedulerEmitter (static) got invalid number of inputs. Expected 3, got " << in.size();
    if (!jcp.is_static && in.size() != 4)
        IE_THROW() << "TileSchedulerEmitter (dynamic) got invalid number of inputs. Expected 4, got " << in.size();
    if (out.size() != in[0] + in[1])
        IE_THROW() << "TileSchedulerEmitter got invalid number of outputs. Expected " << in[0] + in[1] << " , got " << out.size();
    if (body.size() != 2)
        IE_THROW() << "TileSchedulerEmitter got invalid body size, expected 2 (vector & scalar TileEmitter), got " << body.size();
    if (!(std::dynamic_pointer_cast<TileEmitter>(body[0].first) && std::dynamic_pointer_cast<TileEmitter>(body[1].first)))
        IE_THROW() << "TileSchedulerEmitter can contain only TileEmitters inside its body";
}

void TileSchedulerEmitter::emit_static_tiles(const Reg64& reg_inner_amount, const std::vector<Reg64>& data_ptr_regs, size_t vector_size,
                                      const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    // TileAllocatedEmitter is just an alias to perform dynamic_pointer_cast only once and reuse it below several times
    using TileAllocatedEmitter = std::pair<std::shared_ptr<TileEmitter>, const ngraph::snippets::RegInfo&>;
    TileAllocatedEmitter vector_tile {std::dynamic_pointer_cast<TileEmitter>(body[0].first), body[0].second};
    TileAllocatedEmitter scalar_tile {std::dynamic_pointer_cast<TileEmitter>(body[1].first), body[1].second};
    const size_t inner_work_amount = jcp.scheduler_work_amounts[1];
    const size_t outer_work_amount = jcp.scheduler_work_amounts[0];
    auto process_tile =
        [&](const bool evaluate_once, const bool skip_increments, const TileAllocatedEmitter& tile) {
            // If Tile is evaluated only once, then we can emit its body directly and skip work_amount decrements and checks
            if (evaluate_once) {
                tile.first->emit_body(vec_pool, gpr_pool);
                if (!skip_increments)
                    tile.first->emit_ptr_increments_static(data_ptr_regs);
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
        const bool skip_increments = outer_work_amount == 1 && inner_work_amount == vector_size;
        // Need to set proper work amount for inner tiles if evaluated multiple times
        if (!vector_evaluate_once)
            h->mov(reg_inner_amount, inner_work_amount);
        process_tile(vector_evaluate_once, skip_increments, vector_tile);
    }
    if (inner_work_amount % vector_size >= 1) {
        bool scalar_evaluate_once = inner_work_amount % vector_size < 2;
        if (!scalar_evaluate_once) {
            // vector_tile is not executed, work_amount is not set
            if (inner_work_amount < vector_size) {
                h->mov(reg_inner_amount, inner_work_amount);
                // vector_tile is executed, but work_amount is neither set nor decremented appropriately.
            } else if (vector_evaluate_once) {
                h->mov(reg_inner_amount, inner_work_amount - vector_size);
            }
            // else: vector_tile is executed multiple times, so work_amount is already set
        }
        const bool skip_increments = outer_work_amount == 1 && inner_work_amount % vector_size == 1;
        process_tile(scalar_evaluate_once, skip_increments, scalar_tile);
    }
}

void TileSchedulerEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& vec_pool,
                                     const std::vector<size_t>& gpr_pool,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    if (jcp.is_static)
        emit_static_impl(in, out, vec_pool, gpr_pool, emit_context);
    else
        emit_dynamic_impl(in, out, vec_pool, gpr_pool, emit_context);
}

void TileSchedulerEmitter::emit_static_impl(const std::vector<size_t>& in,
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
    const size_t outer_work_amount = jcp.scheduler_work_amounts[0];
    if (outer_work_amount == 1) {
        // emit code directly without looping over external dim
        emit_static_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);
    } else if (outer_work_amount > 1) {
        // We need to create a Loop in this case
        h->mov(reg_outer_amount, outer_work_amount);
        h->L(for_body);
        {
            emit_static_tiles(reg_inner_amount, data_ptr_regs, vector_size, vec_pool, local_gpr_pool);

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

void TileSchedulerEmitter::emit_dynamic_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            const std::vector<size_t>& vec_pool,
                                            const std::vector<size_t>& gpr_pool,
                                            const ov::intel_cpu::emitter_context *emit_context) const {
    const size_t num_inputs = in[0];
    const size_t num_outputs = in[1];
    const size_t vector_size = in[2];
    const size_t num_params = num_inputs + num_outputs;

    const auto& data_ptr_reg_idxs(out);
    std::vector<Reg64> data_ptr_regs(data_ptr_reg_idxs.size());
    std::transform(data_ptr_reg_idxs.begin(), data_ptr_reg_idxs.end(), data_ptr_regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});

    Reg64 reg_const_params = Reg64(static_cast<int>(in[3]));
    // todo: this limitation could be removed if we use Reg32 to store work_amounts (which is more than enough),
    //  since at least one Reg64 (reg_indexes spared in the Kernel) is guaranteed to be in the pool
    if (gpr_pool.size() < 2)
        IE_THROW() << "Dynamic Tile Scheduler needs at least two spare gpr regs to operate.";
    auto local_gpr_pool = gpr_pool;
    Reg64 reg_outer_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    Reg64 reg_inner_amount = Reg64(static_cast<int>(local_gpr_pool.back()));
    local_gpr_pool.pop_back();
    using TileAllocatedEmitter = std::pair<std::shared_ptr<TileEmitter>, const ngraph::snippets::RegInfo&>;
    TileAllocatedEmitter vector_tile {std::dynamic_pointer_cast<TileEmitter>(body[0].first), body[0].second};
    TileAllocatedEmitter scalar_tile {std::dynamic_pointer_cast<TileEmitter>(body[1].first), body[1].second};
    auto emit_tiles = [&]() {
        // the minimal requirement is that tile (vector or scalar) is emitted only if it has some work to do (>= 1 iterations)
        auto process_tile =
            [&](const size_t tile_increment, const TileAllocatedEmitter& tile) {
                    Label tile_end;
                    h->cmp(reg_inner_amount, tile_increment);
                    h->jl(tile_end, CodeGenerator::T_NEAR);
                    std::vector<size_t> in_regs, out_regs;
                    std::tie(in_regs, out_regs) = tile.second;
                    // pass work_amount reg to Tile
                    in_regs.push_back(static_cast<size_t>(reg_inner_amount.getIdx()));
                    in_regs.push_back(static_cast<size_t>(reg_const_params.getIdx()));
                    for (const auto& reg : data_ptr_regs)
                        out_regs.emplace_back(reg.getIdx());
                    tile.first->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
                    h->L(tile_end);
            };
        h->mov(reg_inner_amount, h->ptr[reg_const_params + GET_OFF(scheduler_work_amounts) + sizeof(size_t)]);
        process_tile(vector_size, vector_tile);
        process_tile(1, scalar_tile);
    };
    Label for_body, single_outer_tile, end;
    {
        h->mov(reg_outer_amount, h->ptr[reg_const_params + GET_OFF(scheduler_work_amounts)]);
        // We don't need to apply scheduler offsets, or update reg_outer_amount in case of outer WA == 1
        h->cmp(reg_outer_amount, 1);
        h->je(single_outer_tile, CodeGenerator::T_NEAR);
        //
        h->L(for_body);
        {
            emit_tiles();

            // Todo: Load and Store emitters are currently implemented so they ALWAYS increment appropriate pointers
            //   after reading/writing. This might be a problem if we need to read the same data multiple times (broadcasting shapes).
            //   To overcome this limitation, we add appropriate negative offsets if necessary.
            for (auto i = 0; i < num_params; i++) {
                // NB! many scheduler offsets are zero
                h->add(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(scheduler_offsets) + i * sizeof(int64_t)]);
            }
            // Note that outer dimensions are always incremented by 1 (outer tiles are always scalar)
            h->sub(reg_outer_amount, 1);
            h->cmp(reg_outer_amount, 1);
            h->jge(for_body, CodeGenerator::T_NEAR);
            h->jmp(end, CodeGenerator::T_NEAR);
        }
        h->L(single_outer_tile);
        {
            // emit code directly without looping over external dim and applying scheduler offsets
            emit_tiles();
        }
        h->L(end);
    }
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
    size_t num_dynamic_inputs = 0;
    const bool has_dynamic_dims = std::any_of(io_dims.begin(), io_dims.end(),
                                              [](size_t x) {return x == Subgraph::DYNAMIC_DIMENSION;});
    for (size_t i = 0; i < io_dims.size(); i ++) {
        // If a last dim is static, but == 1 and there are some dynamic inputs as well,
        // then treat the dim as dynamic, since we'll now whether it's broadcasted only at runtime
        if (io_dims[i] == Subgraph::DYNAMIC_DIMENSION || (io_dims[i] == 1 && has_dynamic_dims)) {
            dynamic_dims_idx.push_back(i);
            if (i < num_inputs)
                num_dynamic_inputs++;
        } else {
            static_dims_idx.push_back(i);
        }
    }
    dynamic_increments.resize(dynamic_dims_idx.size());
    dynamic_broadcasting.resize(num_dynamic_inputs);
    // zero in io_dims indicates dynamic dimension
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
    //todo: if one of the uppermost dimensions is dynamic (batch for example), node is still considered to be dynamic
    // and evaluates dynamic pipeline. Hence dynamic_dims_idx may be empty, but in.size() still == 2. Fix this in future.
    //  if ((dynamic_dims_idx.empty() && in.size() != 1)  || (!dynamic_dims_idx.empty() && in.size() !=2))
    if (in.size() != 1  && in.size() !=2)
        IE_THROW() << "TileEmitter got invalid number of inputs.";
    if (out.size() != io_dims.size())
        IE_THROW() << "TileEmitter got invalid number of outputs. Expected " << io_dims.size() << " , got " << out.size();
}

void TileEmitter::emit_body(const std::vector<size_t>& vec_pool, const std::vector<size_t>& gpr_pool) const {
    for (auto& code : body)
        code.first->emit_code(code.second.first, code.second.second, vec_pool, gpr_pool);
}

void TileEmitter::emit_ptr_increments_static(const std::vector<Reg64>& data_ptr_regs) const {
    // note that master_shape_last_dim could be equal to Subgraph::DYNAMIC_DIMENSION for dynamic case
    auto master_shape_last_dim = *std::max_element(io_dims.begin(), io_dims.end());
    for (const auto& idx : static_dims_idx) {
        // increment only inputs that are not broadcasted
        if (io_dims[idx] != 1 || master_shape_last_dim == 1)
            h->add(data_ptr_regs[idx], increment * io_data_size[idx]);
    }
}

void TileEmitter::emit_ptr_increments_dynamic(const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    emit_ptr_increments_static(data_ptr_regs);
    const size_t tile_type_offset = increment > 1 ? GET_OFF(vector_tile_increments) : GET_OFF(scalar_tile_increments);
    for (size_t i = 0; i < dynamic_dims_idx.size(); i++) {
        auto idx = dynamic_dims_idx[i];
        h->add(data_ptr_regs[idx], h->ptr[reg_const_params + tile_type_offset + idx * sizeof(int64_t)]);
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void TileEmitter::set_increments_and_broadcast_inputs(const Reg64& reg_const_params, const std::vector<Reg64> &data_ptr_regs) const {
        using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                             Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
        auto Vmm_tmp = Vmm(0);
        for (size_t i = 0; i < dynamic_dims_idx.size(); i++) {
            auto idx = dynamic_dims_idx[i];
            const auto& data_ptr_reg = data_ptr_regs[idx];
            // todo: we can store dynamic broadcasting info only for dynamic inputs (not for all, like we do now)
            h->cmp(h->byte[reg_const_params + GET_OFF(broadcasting_mask) + idx * sizeof(bool)], 0);
            Label no_broadcasting;
            h->je(no_broadcasting, CodeGenerator::T_SHORT);
            // Both inputs and outputs can be dynamic, but only inputs could be physically broadcasted
            // Physical broadcasting is only required for vector tiles
            if (idx < num_inputs && increment != 1) {
                h->push(data_ptr_reg);
                h->uni_vbroadcastss(Vmm_tmp, h->ptr[data_ptr_reg]);
                h->mov(data_ptr_reg, h->ptr[reg_const_params + GET_OFF(broadcasting_scratchpad)]);
                h->add(data_ptr_reg, i * increment *  io_data_size[idx]);
                // note that we use data_ptr_reg directly without h->rip
                h->uni_vmovups(h->ptr[data_ptr_reg], Vmm_tmp);
            }
            h->L(no_broadcasting);
        }
}

void TileEmitter::cleanup_broadcasting(const Reg64& reg_const_params, const std::vector<Reg64> &data_ptr_regs) const {
    if (increment == 1)
        return;
    for (int i = static_cast<int>(dynamic_dims_idx.size()) - 1; i >= 0; i--) {
        const auto& idx = dynamic_dims_idx[i];
        if (idx >= num_inputs)
            continue;
        // todo: we can store dynamic broadcasting info only for dynamic inputs (not for all, like we do now)
        Label no_broadcasting;
        h->cmp(h->byte[reg_const_params + GET_OFF(broadcasting_mask) + idx * sizeof(bool)], 0);
        h->je(no_broadcasting, CodeGenerator::T_SHORT);
        h->pop(data_ptr_regs[idx]);
        h->L(no_broadcasting);
    }
}

void TileEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& vec_pool,
                            const std::vector<size_t>& gpr_pool,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    Reg64 work_amount = Reg64(static_cast<int>(in[0]));
    Reg64 reg_const_params;
    // todo: unify interface for static & dynamic calls for TileEmitter?
    // There is 1 arg for the static case, so we can assign any reg to reg_const_params, since it won't be really used.
    // Anyway, try to assign a reg from the pool to prevent possible work_amount corruption
    if (dynamic_dims_idx.empty()) {
        reg_const_params = gpr_pool.empty() ? work_amount : Reg64(gpr_pool.back());
    } else {
        reg_const_params = Reg64(static_cast<int>(in[1]));
    }
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(out, data_ptr_regs);
    switch (host_isa_) {
        case dnnl::impl::cpu::x64::sse41:
            set_increments_and_broadcast_inputs<dnnl::impl::cpu::x64::sse41>(reg_const_params, data_ptr_regs);
            break;
        case dnnl::impl::cpu::x64::avx2:
            set_increments_and_broadcast_inputs<dnnl::impl::cpu::x64::avx2>(reg_const_params, data_ptr_regs);
            break;
        case dnnl::impl::cpu::x64::avx512_core:
            set_increments_and_broadcast_inputs<dnnl::impl::cpu::x64::avx512_core>(reg_const_params, data_ptr_regs);
            break;
        default:
            IE_THROW() << "unsupported isa: " << host_isa_;
    }
    Label for_body;
    // Note that:
    // * Work amount must be set by TileScheduler that executes Tiles
    // * TileScheduler executes Tile only if it has to perform >= 1 iterations
    h->L(for_body);
    emit_body(vec_pool, gpr_pool);
    emit_ptr_increments_dynamic(reg_const_params, data_ptr_regs);
    h->sub(work_amount, increment);
    h->cmp(work_amount, increment);
    h->jge(for_body, CodeGenerator::T_NEAR);
    cleanup_broadcasting(reg_const_params, data_ptr_regs);
}

TileBeginEmitter::TileBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    tileBegin = ov::as_type_ptr<ngraph::snippets::op::TileBegin>(n);
    if (!tileBegin)
        IE_THROW() << "TileBeginEmitter invoked with invalid op argument";
    const auto& target_inputs = tileBegin->output(tileBegin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        IE_THROW() << "TileBeginEmitter invoked with invalid op argument";
    const auto tileEnd = ov::as_type_ptr<ngraph::snippets::op::TileEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!tileEnd)
        IE_THROW() << "TileBeginEmitter invoked with invalid configuration: the last output must be TileEnd";
    const auto io_size = tileBegin->get_input_size() + tileEnd->get_output_size();
    if (tileBegin->get_finalization_offsets().size() != io_size)
        IE_THROW() << "TileBeginEmitter got invalid op configuration: finalization_offsets size is incorrect";
    if (tileBegin->get_apply_increment().size() != io_size)
        IE_THROW() << "TileBeginEmitter got invalid op configuration: apply_increment size is incorrect";
    num_inputs = tileBegin->get_input_size();
    num_outputs = tileEnd->get_output_size();
    increment = tileBegin->get_increment();
    work_amount = tileBegin->get_work_amount();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    /*
    // todo: add checks on work_amount vs increment consistency + checks on work_amount vs max(last_dim) consistence
    //  probably better to implement them in Tile* constructors
    for (int i = 0; i < num_inputs; i++) {
        // todo: we can take the whole partial shape in future, since it should contain only one dim
        const auto& last_dim = *(tileBegin->get_input_partial_shape(i).rbegin());
        io_dims.push_back(last_dim.is_static() ? last_dim.get_length() : 0);
        io_data_size.push_back(tileBegin->get_input_element_type(i).size());
    }
    for (int i = 0; i < num_outputs; i++) {
        // todo: we can take the whole partial shape in future, since it should contain only one dim
        const auto& last_dim = *(tileEnd->get_output_partial_shape(i).rbegin());
        io_dims.push_back(last_dim.is_static() ? last_dim.get_length() : 0);
        io_data_size.push_back(tileBegin->get_input_element_type(i).size());
    }
    size_t num_dynamic_inputs = 0;
    const bool has_dynamic_dims = std::any_of(io_dims.begin(), io_dims.end(), [](size_t x) {return x == 0;});
    for (size_t i = 0; i < io_dims.size(); i ++) {
        // If a last dim is static, but == 1 and there are some dynamic inputs as well,
        // then treat the dim as dynamic, since we'll now whether it's broadcasted only at runtime
        if (io_dims[i] == 0 || (io_dims[i] == 1 && has_dynamic_dims)) {
            dynamic_dims_idx.push_back(i);
            if (i < num_inputs)
                num_dynamic_inputs++;
        } else {
            static_dims_idx.push_back(i);
        }
    }
    dynamic_increments.resize(dynamic_dims_idx.size());
    dynamic_broadcasting.resize(num_dynamic_inputs);
    */
}

void TileBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out,
                                 const std::vector<size_t>& pool,
                                 const std::vector<size_t>& gpr,
                                 const ov::intel_cpu::emitter_context *emit_context) const {
    // todo: In dynamic case we will also need to set broadcasting info here
    // todo: skip everything if work_amount == increment
    Reg64 reg_work_amount = Reg64(abi_param2.getIdx());
    Label for_body;
    // save previous register state (if there is an outer loop that uses this reg for example)
    // if work_amount == 0, it means that work_amount was set in the previous tile, so we should not reset it here
    if (work_amount != 0) {
        h->push(reg_work_amount);
        h->mov(reg_work_amount, work_amount);
    }
    // todo fix excessive push-pop with an appropriate gpr assign_registers pass
    // h->L(for_body);
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    tileBegin->begin_address = h->getCurr();
    tileBegin->input_regs = in;
}

TileEndEmitter::TileEndEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto tileEnd = ov::as_type_ptr<ngraph::snippets::op::TileEnd>(n);
    if (!tileEnd)
        IE_THROW() << "TileEndEmitter invoked with invalid op argument";
    tileBegin = ov::as_type_ptr<ngraph::snippets::op::TileBegin>(
            tileEnd->get_input_source_output(tileEnd->get_input_size() - 1).get_node_shared_ptr());
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!tileBegin)
        IE_THROW() << "TileEndEmitter invoked with invalid configuration: the last arg must be TileBegin";
    num_inputs = tileBegin->get_input_size();
    num_outputs = tileEnd->get_output_size();
    increment = tileBegin->get_increment();
    work_amount = tileBegin->get_work_amount();
    // todo: add checks on work_amount vs increment consistency + checks on work_amount vs max(last_dim) consistence
    //  probably better to implement them in Tile* constructors
    for (int i = 0; i < num_inputs; i++) {
        // todo: we can take the whole partial shape in future, since it should contain only one dim
        const auto& relevant_dim = tileBegin->get_input_partial_shape(i)[tileBegin->get_dimension()];
        io_dims.push_back(relevant_dim.is_static() ? relevant_dim.get_length() : Subgraph::DYNAMIC_DIMENSION);
        io_data_size.push_back(tileBegin->get_input_element_type(i).size());
    }
    for (int i = 0; i < num_outputs; i++) {
        // todo: we can take the whole partial shape in future, since it should contain only one dim
        const auto& relevant_dim = tileEnd->get_output_partial_shape(i)[tileEnd->get_dimension()];
        io_dims.push_back(relevant_dim.is_static() ? relevant_dim.get_length() : Subgraph::DYNAMIC_DIMENSION);
        io_data_size.push_back(tileBegin->get_input_element_type(i).size());
    }
    size_t num_dynamic_inputs = 0;
    const bool has_dynamic_dims = std::any_of(io_dims.begin(), io_dims.end(), [](size_t x) {return x == Subgraph::DYNAMIC_DIMENSION;});
    for (size_t i = 0; i < io_dims.size(); i ++) {
        // If a last dim is static, but == 1 and there are some dynamic inputs as well,
        // then treat the dim as dynamic, since we'll now whether it's broadcasted only at runtime
        if (io_dims[i] == Subgraph::DYNAMIC_DIMENSION || (io_dims[i] == 1 && has_dynamic_dims)) {
            dynamic_dims_idx.push_back(i);
            if (i < num_inputs)
                num_dynamic_inputs++;
        } else {
            static_dims_idx.push_back(i);
        }
    }
    dynamic_increments.resize(dynamic_dims_idx.size());
    dynamic_broadcasting.resize(num_dynamic_inputs);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void TileEndEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileEndEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out,
                                 const std::vector<size_t>& pool,
                                 const std::vector<size_t>& gpr,
                                 const ov::intel_cpu::emitter_context *emit_context) const {
    std::vector<size_t> data_ptr_reg_idxs(tileBegin->input_regs);
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(out.begin(), out.end(), std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(abi_param2.getIdx());
    // Nothing to do in this case
    // todo: who will increment if there is non-zero outer tile?
//    if (work_amount == increment)
//        return;
    const auto& apply_increments = tileBegin->get_apply_increment();
    if (apply_increments.size() != data_ptr_regs.size())
        IE_THROW() << "Inconsistent apply increments and data_ptr_regs size";
    for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (apply_increments[idx])
            h->add(data_ptr_regs[idx], increment * io_data_size[idx]);
    }
    h->sub(reg_work_amount, increment);
    h->cmp(reg_work_amount, increment);
    h->jge(tileBegin->begin_address);

    const auto& finalization_offsets = tileBegin->get_finalization_offsets();
    if (finalization_offsets.size() != data_ptr_regs.size())
        IE_THROW() << "Inconsistent finalization offsets and data_ptr_regs size";
    for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
    if (tileBegin->get_work_amount() != 0) {
        // restore reg state if we've changed it before
        h->pop(reg_work_amount);
    }
//    cleanup_broadcasting(reg_const_params, data_ptr_regs);
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
