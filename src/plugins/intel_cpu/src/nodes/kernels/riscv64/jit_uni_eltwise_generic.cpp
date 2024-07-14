// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

#include "emitters/plugin/riscv64/jit_add_emitter.hpp"
#include "emitters/plugin/riscv64/jit_divide_emitter.hpp"
#include "emitters/plugin/riscv64/jit_multiply_emitter.hpp"
#include "emitters/plugin/riscv64/jit_power_static_emitter.hpp"
#include "emitters/plugin/riscv64/jit_subtract_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

void jit_uni_eltwise_kernel::operator()(
    const node::jit_eltwise_call_args_ptrs* const_args,
    const jit_eltwise_call_args_indexes* indexes) {
    assert(ker_);
    ker_(const_args, indexes);
}

jit_uni_eltwise_generic::jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                                 const std::vector<EltwiseData>& eltwise_data,
                                                 const std::vector<ov::intel_cpu::Type>& ops_list,
                                                 const dnnl::post_ops& post_ops) :
                                                 jit_uni_eltwise_kernel(jep),
                                                 jit_generator(),
                                                 eltwise_data_(eltwise_data),
                                                 ops_list_(ops_list),
                                                 post_ops_(post_ops) {
}

void jit_uni_eltwise_generic::generate() {
    preamble();

    auto const exec_prc = ov::element::f32;
    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    size_t input_count = eltwise_emitter->get_inputs_count();
    size_t max_aux_count = eltwise_emitter->get_aux_vecs_count();
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        const auto emitter = create_eltwise_emitter(eltwise_data_[i], exec_prc);
        post_op_emitters.push_back(emitter);

        input_count += emitter->get_inputs_count() - 1;
        max_aux_count = std::max(max_aux_count, emitter->get_aux_vecs_count());
    }

    const auto getLmul = [](const size_t reg_count) {
        const auto multiplier = static_cast<size_t>(32 / reg_count);
        if (multiplier <= 1) {
            return LMUL::m1;
        } else if ((multiplier == 2) || (multiplier == 3)) {
            return LMUL::m2;
        } else if ((4 <= multiplier) && (multiplier <= 7)) {
            return LMUL::m4;
        } else {
            return LMUL::m8;
        }
    };

    const size_t vec_registers_count = input_count + max_aux_count + 1;
    const size_t vmm_dst_idx = vec_registers_count - 1;
    const LMUL lmul = getLmul(vec_registers_count);

    const auto &jep = jep_;

    Reg param1 = Xbyak_riscv::a0;
    Reg param2 = Xbyak_riscv::a1;

    Reg reg_post_op_ptrs = t0;
    Reg start_to_offsets = reg_post_op_ptrs;

    Reg reg_const_params = param1;
    Reg reg_indexes = param2;

    const int offset_count = jep.input_size - 1;

    // ptrs initializing
    if (jep.use_runtime_ptrs) {
        for (size_t i = 0; i < jep.inputs_number; i++) {
            ld(start_to_offsets, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_offsets) + i * sizeof(size_t)));
            ld(get_src_reg(i), reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[0]) + i * sizeof(size_t)));

            Reg offset_reg = get_aux_gpr(0);
            Reg index_reg = get_aux_gpr(1);
            for (int j = 0; j < offset_count; j++) {
                ld(offset_reg, start_to_offsets, static_cast<int32_t>(j * sizeof(size_t)));
                ld(index_reg, reg_indexes, static_cast<int32_t>(j * sizeof(size_t)));
                mul(offset_reg, offset_reg, index_reg);
                add(get_src_reg(i), offset_reg, get_src_reg(i));
            }
        }

        ld(start_to_offsets, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_offsets)));
        ld(reg_dst, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr)));
        Reg offset_reg = get_aux_gpr(0);
        Reg index_reg = get_aux_gpr(1);
        for (int j = 0; j < offset_count; j++) {
            ld(offset_reg, start_to_offsets, static_cast<int32_t>(j * sizeof(size_t)));
            ld(index_reg, reg_indexes, static_cast<int32_t>(j * sizeof(size_t)));
            mul(offset_reg, offset_reg, index_reg);
            add(reg_dst, offset_reg, reg_dst);
        }

        ld(reg_work_amount, reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, work_amount)));
    } else {
        auto init_ptrs_with_offsets = [this, offset_count, param2](Reg pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    Reg offset_reg(get_aux_gpr(0));
                    li(offset_reg, static_cast<int>(offsets[j]));
                    Reg index_reg(get_aux_gpr(1));
                    ld(index_reg, param2, static_cast<int32_t>(j * sizeof(size_t)));

                    mul(offset_reg, offset_reg, index_reg);
                    add(pointer, pointer, offset_reg);
                }
            }
        };

        for (size_t i = 0; i < jep.inputs_number; i++) {
            const auto src_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t));
            ld(get_src_reg(i), param1, src_offset);
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
        }

        const auto dst_offset = static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr));
        ld(reg_dst, reg_const_params, dst_offset);
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        addi(reg_work_amount, x0, jep.work_amount); // li
    }

    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            // TODO: move outside of loop
            vsetvli(t0, reg_work_amount, SEW::e32, lmul);

            flw(f0, get_src_reg(i));
            const auto vmm_reg = get_vmm_reg(i, lmul);
            vxor_vv(vmm_reg, vmm_reg, vmm_reg);
            vfadd_vf(vmm_reg, vmm_reg, f0);
        }
    }

    size_t min_src_size = jep.dst_size;
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] != 1)
            min_src_size = std::min(min_src_size, jep.src_size[i]);
    }
    if (jep_.oc_size > 1)
        min_src_size = std::min(min_src_size, jep_.oc_size);

    if (min_src_size != jep.dst_size) {
        bool is_valid_configuration = true;
        if (jep.dst_size % min_src_size != 0)
            is_valid_configuration = false;

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                is_valid_configuration = false;
        }

        if (jep.oc_size > 1 && jep.oc_size != min_src_size && jep.oc_size != jep.dst_size)
            is_valid_configuration = false;

        if (!is_valid_configuration)
            OPENVINO_THROW("Eltwise jitter has invalid configuration for Eltwise node");

        Label unroll_loop_label;
        Label unroll_loop_label2;
        Label unroll_loop_end_label;

        L(unroll_loop_label);
        {
            const size_t loop_step = min_src_size;
            // TODO: use register, made it once
            const auto reg_loop_step = t5;
            li(reg_loop_step, loop_step);

            blt(reg_work_amount, reg_loop_step, unroll_loop_end_label);

            // aux gpr is used by emitters
            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    add(get_aux_gpr_kernel(i), x0, get_src_reg(i));
                }
            }

            li(t1, static_cast<int>(min_src_size));

            L(unroll_loop_label2);
            {
                vsetvli(t0, t1, SEW::e32, lmul);

                sub(t1, t1, t0);
                slli(t0, t0, 2);

                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        //load_vector(get_vmm_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, false, j * vec_step * jep.src_prc[i].size());
                        vle32_v(get_vmm_reg(i, lmul), get_aux_gpr_kernel(i));
                        add(get_aux_gpr_kernel(i), get_aux_gpr_kernel(i), t0);
                    }
                }

                compute_eltwise_op(lmul, input_count, vmm_dst_idx);

                apply_post_ops(lmul, input_count, vmm_dst_idx);

                vse32_v(get_dst_vmm(vmm_dst_idx, lmul), reg_dst);
                add(reg_dst, reg_dst, t0);

                bnez(t1, unroll_loop_label2);
            }

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] == jep.dst_size) {
                    li(reg_tmp, jep.src_prc[i].size() * min_src_size);
                    add(get_src_reg(i), get_src_reg(i), reg_tmp);
                }
            }

            sub(reg_work_amount, reg_work_amount, reg_loop_step);
            j_(unroll_loop_label);
        }

        L(unroll_loop_end_label);
    }

    if (min_src_size == jep.dst_size) {
        Label main_loop_label;
        Label main_loop_end_label;

        L(main_loop_label);
        {
            vsetvli(t0, reg_work_amount, SEW::e32, lmul);
            sub(reg_work_amount, reg_work_amount, t0);
            slli(t0, t0, 2);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    vle32_v(get_vmm_reg(i, lmul), get_src_reg(i));
                    add(get_src_reg(i), get_src_reg(i), t0);
                }
            }

            compute_eltwise_op(lmul, input_count, vmm_dst_idx);

            apply_post_ops(lmul, input_count, vmm_dst_idx);

            vse32_v(get_dst_vmm(vmm_dst_idx, lmul), reg_dst);
            add(reg_dst, reg_dst, t0);

            bnez(reg_work_amount, main_loop_label);
        }
        L(main_loop_end_label);
    }

    postamble();

    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        post_op_emitters[i]->emit_data();
    }
}

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    ov::intel_cpu::riscv64::jit_generator *host;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host,
                                                                 ctx.opData.alpha,
                                                                 ctx.opData.beta,
                                                                 ctx.opData.gamma,
                                                                 ctx.exec_prc);
    }
};

std::shared_ptr<jit_emitter> jit_uni_eltwise_generic::create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {
        nullptr,
        this,
        data,
        exec_prec
    };

    OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
    OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
    OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
    OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
    OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
    OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));

    if (!ctx.emitter)
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");

    return ctx.emitter;
}

void jit_uni_eltwise_generic::compute_eltwise_op(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx) {
    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_count(); i++) {
        in_idxs.push_back(get_vmm_reg(i, lmul).getIdx());
    }

    std::vector<size_t> aux_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_vecs_count(); i++) {
        aux_idxs.push_back(get_aux_vmm(i, lmul, input_reg_count).getIdx());
    }

    std::vector<size_t> out_idxs;
    out_idxs.push_back(get_dst_vmm(vmm_dst_idx, lmul).getIdx());

    std::vector<size_t> gpr_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_gprs_count(); i++) {
        gpr_idxs.push_back(get_aux_gpr(i).getIdx());
    }

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs, gpr_idxs);
}

void jit_uni_eltwise_generic::apply_post_ops(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx) {
    int input_idx = eltwise_emitter->get_inputs_count();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < ops_list_.size(); i++) {
        if (ops_list_[i] == ov::intel_cpu::Type::Eltwise) {
            std::vector<size_t> in_idxs;
            in_idxs.push_back(get_dst_vmm(vmm_dst_idx, lmul).getIdx());
            for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_count(); j++)
                in_idxs.push_back(get_vmm_reg(input_idx++, lmul).getIdx());

            std::vector<size_t> out_idxs;
            out_idxs.push_back(get_vmm_reg(vmm_dst_idx, lmul).getIdx());

            std::vector<size_t> aux_vmm_idxs;
            for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->get_aux_vecs_count(); j++)
                aux_vmm_idxs.push_back(get_aux_vmm(j, lmul, input_reg_count).getIdx());

            std::vector<size_t> aux_gpr_idxs;
            for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->get_aux_gprs_count(); j++)
                aux_gpr_idxs.push_back(get_aux_gpr(j).getIdx());

            post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_vmm_idxs, aux_gpr_idxs);

            eltwise_post_op_idx++;
        } else if (ops_list_[i] == ov::intel_cpu::Type::FakeQuantize) {
            OPENVINO_THROW("Eltwise jit kernel: FakeQuantize is not supported");
        } else {
            OPENVINO_THROW("Eltwise jit kernel: unexpected operation type");
        }
    }
}

uint32_t jit_uni_eltwise_generic::lmul2int(const LMUL lmul) {
    switch (lmul) {
        case LMUL::m1: {
            return 1;
        }
        case LMUL::m2: {
            return 2;
        }
        case LMUL::m4: {
            return 4;
        }
        case LMUL::m8: {
            return 8;
        }
        default: {
            OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(static_cast<uint32_t>(lmul)));
        }
    }
}

Reg jit_uni_eltwise_generic::get_src_reg(uint32_t idx) {
    if (idx > MAX_ELTWISE_INPUTS) {
        OPENVINO_THROW("source vector ptr register " + std::to_string(idx) + " is not supported");
    }
    return Reg(18 + idx);
}

Reg jit_uni_eltwise_generic::get_aux_gpr(const uint32_t idx) {
    const uint32_t begin_idx = 28;
    const uint32_t end_idx = 31;
    if ((begin_idx + idx) > end_idx) {
        OPENVINO_THROW("aux gpr register " + std::to_string(idx) + " is not supported");
    }

    return Reg(begin_idx + idx);
}

Reg jit_uni_eltwise_generic::get_aux_gpr_kernel(const uint32_t idx) {
    const uint32_t begin_idx = 24;
    const uint32_t end_idx = 26;
    if ((begin_idx + idx) > end_idx) {
        OPENVINO_THROW("aux gpr register kernel " + std::to_string(idx) + " is not supported");
    }

    return Reg(begin_idx + idx);
}

VReg jit_uni_eltwise_generic::get_vmm_reg(const uint32_t idx, const LMUL lmul) {
    const uint32_t physical_idx = idx * lmul2int(lmul);
    if (physical_idx > 31) {
        OPENVINO_THROW("source vector register " + std::to_string(idx) + " (" + std::to_string(physical_idx) + ") is not supported");
    }
    return VReg(physical_idx);
}

VReg jit_uni_eltwise_generic::get_dst_vmm(const uint32_t idx, const LMUL lmul) {
    const uint32_t physical_idx = idx * lmul2int(lmul);
    if (physical_idx > 31) {
        OPENVINO_THROW("destination vector register " + std::to_string(idx) + " (" + std::to_string(physical_idx) + ") is not supported");
    }
    return VReg(physical_idx);
}

//SReg jit_uni_eltwise_generic::get_scl_reg(const uint32_t idx) {
//    if (idx > MAX_ELTWISE_INPUTS) {
//        OPENVINO_THROW("source scalar register " + std::to_string(idx) + " is not supported");
//    }
//    return SReg(0 + idx);
//}

VReg jit_uni_eltwise_generic::get_aux_vmm(const uint32_t idx, const LMUL lmul, const uint32_t start_idx) {
    const uint32_t phisical_idx =(start_idx + idx) * lmul2int(lmul);
    if (phisical_idx > 31) {
        OPENVINO_THROW("aux vector register " + std::to_string(idx) + " is not supported");
    }
    return VReg(phisical_idx);
}

void jit_uni_eltwise_generic::load_vector(const VReg& data,
                                          const Reg& ptr_reg,
                                          const ov::element::Type& src_prc,
                                          const ov::element::Type& dst_prc,
                                          const bool broadcast,
                                          const int32_t ptr_offset) {
}

void jit_uni_eltwise_generic::store_vector(const Reg& ptr,
                                           const VReg& data,
                                           const ov::element::Type& src_prc,
                                           const ov::element::Type& dst_prc,
                                           const int32_t ptr_offset) {
}

namespace {
template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<std::vector<element::Type>> &precisions) {
        precisions = T::get_supported_precisions();
    }
};
}

ov::element::Type eltwise_precision_helper::get_precision(const size_t inputs_number,
                                                          const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                                          const std::vector<EltwiseData>& eltwise_data) {
    ov::element::Type exec_prc = ov::element::undefined;
    return exec_prc;
}

std::set<std::vector<element::Type>> eltwise_precision_helper::get_supported_precisions(const Algorithm& algo) {
    std::set<std::vector<element::Type>> precisions;

    OV_SWITCH(intel_cpu, SupportedPrecisions, precisions, algo,
              OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
              OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
              OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
              OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
              OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));
    if (precisions.empty())
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");

    return precisions;
}

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
