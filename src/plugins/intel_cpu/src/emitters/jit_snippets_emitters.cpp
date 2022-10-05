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
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], i * offsetRank, reg_tmp);
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
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, local_gpr_pool);
    }
    h->postamble();
}


TileBeginEmitter::TileBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    tile_begin = ov::as_type_ptr<ngraph::snippets::op::TileBegin>(n);
    if (!tile_begin)
        IE_THROW() << "TileBeginEmitter invoked with invalid op argument";
    const auto& target_inputs = tile_begin->output(tile_begin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        IE_THROW() << "TileBeginEmitter invoked with invalid configuration: the last output must have exactly one input attached";
    const auto tile_end = ov::as_type_ptr<ngraph::snippets::op::TileEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!tile_end)
        IE_THROW() << "TileBeginEmitter invoked with invalid configuration: the last output must be TileEnd";
    work_amount = tile_begin->get_work_amount();
    evaluate_once = tile_begin->get_evaluate_once();
    reuse_work_amount_reg = tile_begin->reuse_work_amount_reg;
    num_inputs = tile_begin->get_output_size() - 1;
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void TileBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void TileBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out,
                                        const std::vector<size_t> &pool,
                                        const std::vector<size_t> &gpr) const {
    if (in.size() != num_inputs)
        IE_THROW() << "Invalid inputs size: expected " << num_inputs << " got " << in.size();
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
    if (!evaluate_once && !reuse_work_amount_reg) {
        h->push(reg_work_amount);
        h->mov(reg_work_amount, work_amount);
    }
    // todo fix excessive push-pop with an appropriate gpr assign_registers pass
    // h->L(for_body);
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    tile_begin->begin_address = h->getCurr();
    tile_begin->input_regs = in;
}

TileEndEmitter::TileEndEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    tile_end = ov::as_type_ptr<ngraph::snippets::op::TileEnd>(n);
    if (!tile_end)
        IE_THROW() << "TileEndEmitter invoked with invalid op argument";
    tile_begin = tile_end->get_tile_begin();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!tile_begin)
        IE_THROW() << "TileEndEmitter invoked with invalid configuration: the last arg must be TileBegin";
    // Note that 1 edge connects TileBegin and TileEnd
    num_inputs = tile_begin->get_output_size() - 1;
    num_outputs = tile_end->get_input_size() - 1;
    increment = tile_end->get_increment();
    work_amount = tile_end->get_work_amount();
    apply_increments = tile_end->get_apply_increment();
    finalization_offsets = tile_end->get_finalization_offsets();
    evaluate_once = tile_end->get_evaluate_once();
    reuse_work_amount_reg = tile_end->reuse_work_amount_reg;
    for (int i = 0; i < num_inputs; i++)
        io_data_size.push_back(tile_begin->get_input_element_type(i).size());
    for (int i = 0; i < num_outputs; i++)
        io_data_size.push_back(tile_end->get_input_element_type(i).size());
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void TileEndEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}


void TileEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool,
                                       const std::vector<size_t> &gpr) const {
    if (tile_begin->input_regs.size() != num_inputs)
        IE_THROW() << "Invalid tile_begin->input_regs size: expected " << num_inputs << " got " << tile_begin->input_regs.size();
    if (out.size() != num_outputs)
        IE_THROW() << "Invalid number of out arguments: expected " << num_outputs << " got " << out.size();
    const auto io_size = num_inputs + num_outputs;
    if (apply_increments.size() != io_size)
        IE_THROW() << "Invalid apply_increments size: expected " << io_size << " got " << apply_increments.size();
    if (finalization_offsets.size() != io_size)
        IE_THROW() << "Invalid finalization_offsets size: expected: " << io_size << " got " << finalization_offsets.size();
}

void TileEndEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out,
                                 const std::vector<size_t>& pool,
                                 const std::vector<size_t>& gpr,
                                 const ov::intel_cpu::emitter_context *emit_context) const {
    std::vector<size_t> data_ptr_reg_idxs(tile_begin->input_regs);
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(out.begin(), out.end(), std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(abi_param2.getIdx());
    if (!evaluate_once) {
        for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (apply_increments[idx])
                h->add(data_ptr_regs[idx], increment * io_data_size[idx]);
        }
        h->sub(reg_work_amount, increment);
        h->cmp(reg_work_amount, increment);
        h->jge(tile_begin->begin_address);
    }

    for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
    if (!evaluate_once && !reuse_work_amount_reg) {
        // restore reg state if we've changed it before
        h->pop(reg_work_amount);
    }
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
