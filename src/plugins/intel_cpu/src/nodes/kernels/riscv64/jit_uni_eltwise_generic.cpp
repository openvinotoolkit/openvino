// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

#include "emitters/plugin/riscv64/jit_eltwise_emitters.hpp"


namespace ov::intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) offsetof(jit_eltwise_call_args_ptrs, field)

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
jit_uni_eltwise_generic<isa>::jit_uni_eltwise_generic(jit_eltwise_params jep, std::vector<EltwiseData> eltwise_data)
    : jit_uni_eltwise_kernel(std::move(jep)),
      jit_generator(),
      eltwise_data_(std::move(eltwise_data)) {}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::generate() {
    static const std::vector<element::Type> exec_precisions_priority = { ov::element::i8, ov::element::u8, element::i32, element::f32 };
    auto const exec_prc =
        eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_, exec_precisions_priority);

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    }

    // Labels in Xbyak for RISC-V work with absolute addresses which are defined during call `L(label)`:
    // If we store label to the GPR and only then call `L(label)` to define data section,
    // during kernel compilation null-address will be stored to this GPRs since address will be inited later.
    // To use data section, we define it before kernel execution.
    data_section_address = const_cast<uint8_t*>(getCurr());
    emit_data();

    // After data section, we define code section
    code_section_address = const_cast<uint8_t*>(getCurr());
    preamble();

    const auto& jep = jep_;
    const int offset_count = jep.input_size - 1;

    // ptrs initializing
    if (jep.use_runtime_ptrs) {
        auto init_ptrs_with_offsets = [&](Reg reg) {
            for (int j = 0; j < offset_count; j++) {
                ld(reg_tmp_0, reg_offsets, static_cast<int>(j * sizeof(size_t)));
                ld(reg_tmp_1, reg_indexes, static_cast<int>(j * sizeof(size_t)));
                mul(reg_tmp_0, reg_tmp_0, reg_tmp_1);
                add(reg, reg, reg_tmp_0);
            }
        };

        for (size_t i = 0; i < jep.inputs_number; i++) {
            ld(reg_offsets, reg_const_params, GET_OFF(src_offsets) + i * sizeof(size_t));
            ld(src_gpr(i), reg_const_params, GET_OFF(src_ptr[0]) + i * sizeof(size_t));
            init_ptrs_with_offsets(src_gpr(i));
        }

        ld(reg_offsets, reg_const_params, GET_OFF(dst_offsets));
        ld(dst_gpr(), reg_const_params, GET_OFF(dst_ptr));
        init_ptrs_with_offsets(dst_gpr());

        ld(reg_work_amount, reg_const_params, GET_OFF(work_amount));
    } else {
        auto init_ptrs_with_offsets = [&](Reg reg, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    uni_li(reg_tmp_0, static_cast<int>(offsets[j]));
                    ld(reg_tmp_1, reg_indexes, static_cast<int>(j * sizeof(size_t)));
                    mul(reg_tmp_0, reg_tmp_0, reg_tmp_1);
                    add(reg, reg, reg_tmp_0);
                }
            }
        };

        for (size_t i = 0; i < jep.inputs_number; i++) {
            ld(src_gpr(i), reg_const_params, GET_OFF(src_ptr[0]) + i * sizeof(size_t));
            init_ptrs_with_offsets(src_gpr(i), jep.src_offsets[i]);
        }

        ld(dst_gpr(), reg_const_params, GET_OFF(dst_ptr));
        init_ptrs_with_offsets(dst_gpr(), jep.dst_offsets);

        uni_li(reg_work_amount, static_cast<int>(jep.work_amount));
    }

    exec_lmul = compute_exec_lmul(exec_prc);
    exec_sew = bytes2sew(exec_prc.size());
    OPENVINO_ASSERT(lmul2float(exec_lmul) <= lmul2float(get_max_lmul(exec_prc)),
                    "Unsupported LMUL configuration for the kernel");

    // We set other values for `current` variables to force first call `update_vlen`
    current_lmul = LMUL::m2;
    current_sew = SEW::e8;

    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            load_vector(i, src_gpr(i), reg_work_amount, jep_.src_prc[i], exec_prc, true);
        }
    }

    size_t min_src_size = jep.dst_size;
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] != 1)
            min_src_size = std::min(min_src_size, jep.src_size[i]);
    }

    if (min_src_size == jep.dst_size) {
        Label loop_begin;
        Label loop_end;

        L(loop_begin);
        {
            beqz(reg_work_amount, loop_end);

            // We should call `update_vlen` on each loop iteration to get correct `reg_vlen` in iterations:
            // `rg_vlen` should be valid in in tail loop
            update_vlen(reg_work_amount, exec_sew, exec_lmul, true);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    load_vector(i, src_gpr(i), reg_work_amount, jep_.src_prc[i], exec_prc, false);
                }
            }

            compute_eltwise_op();
            apply_post_ops();

            store_vector(reg_work_amount, exec_prc, jep_.dst_prc);

            sub(reg_work_amount, reg_work_amount, reg_vlen);
            bnez(reg_work_amount, loop_begin);
        }
        L(loop_end);
    }

    if (min_src_size != jep.dst_size) {
        bool is_valid_configuration = true;
        if (jep.dst_size % min_src_size != 0)
            is_valid_configuration = false;

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                is_valid_configuration = false;
        }

        OPENVINO_ASSERT(is_valid_configuration, "Eltwise jitter has invalid configuration for Eltwise node");

        Label loop_begin;
        Label loop_end;
        Label inner_loop_begin;

        L(loop_begin);
        {
            beqz(reg_work_amount, loop_end);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    mv(src_aux_gpr(i), src_gpr(i));
                }
            }

            uni_li(reg_loop_step, min_src_size);
            L(inner_loop_begin);
            {
                // We should call `update_vlen` on each loop iteration to get correct `reg_vlen` in iterations:
                // `rg_vlen` should be valid in in tail loop
                update_vlen(reg_loop_step, exec_sew, exec_lmul, true);

                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        load_vector(i, src_aux_gpr(i), reg_loop_step, jep_.src_prc[i], exec_prc, false);
                    }
                }

                compute_eltwise_op();
                apply_post_ops();

                store_vector(reg_loop_step, exec_prc, jep_.dst_prc);

                sub(reg_loop_step, reg_loop_step, reg_vlen);
                bnez(reg_loop_step, inner_loop_begin);
            }

            const auto reg_tmp = reg_loop_step;
            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] == jep.dst_size) {
                    uni_li(reg_tmp, jep.src_prc[i].size() * min_src_size);
                    add(src_gpr(i), src_gpr(i), reg_tmp);
                }
            }

            uni_li(reg_loop_step, min_src_size);
            sub(reg_work_amount, reg_work_amount, reg_loop_step);
            j_(loop_begin);
        }

        L(loop_end);
    }

    postamble();
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::update_vlen(const Xbyak_riscv::Reg& gpr_work_amount, Xbyak_riscv::SEW sew, Xbyak_riscv::LMUL lmul, bool force) {
    if (!force && current_lmul == lmul && current_sew == sew)
        return;

    vsetvli(reg_vlen, gpr_work_amount, sew, lmul);
    current_lmul = lmul;
    current_sew = sew;

    const auto byte_shift = static_cast<const uint32_t>(sew2bytes(sew) >> 1);
    slli(reg_bvlen, reg_vlen, byte_shift);
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::load_vector(size_t vec_idx, const Xbyak_riscv::Reg& gpr_ptr, const Xbyak_riscv::Reg& gpr_work_amount,
                                               const ov::element::Type& src_prc, const ov::element::Type& dst_prc, bool broadcast) {
    // If input data type is I8/U8 and EXEC_LMUL = m1 and EXEC_SEW = e32 (fp32, for example), we have to update parameters of vector registers
    // to load [BVLEN / e32 x LMUL] I8/U8 values to register for the following conversion to exec precision.
    // Since I8/U8 is sew=e8, we should reduce lmul from `EXEC_LMUL = m1` to `EXEC_LMUL / (EXEC_SEW / e8) = m1 / (e32 / e8) = m1/4`.
    // It allow us to load the needed count of I8/U8 values to vector for the further conversion to exec prc.
    const auto needed_lmul = float2lmul(static_cast<float>(src_prc.size()) / static_cast<float>(dst_prc.size()) * lmul2float(exec_lmul));
    const auto needed_sew = bytes2sew(src_prc.size());
    update_vlen(gpr_work_amount, needed_sew, needed_lmul);

    OPENVINO_ASSERT(dst_prc.size() == sew2bytes(exec_sew), "Incompatible execution SEW and dst SEW");
    OPENVINO_ASSERT(one_of(dst_prc, ov::element::f32, ov::element::i32), "Unsupported dst prc");

    switch (src_prc) {
    case ov::element::f32:
    case ov::element::i32: {
        if (broadcast) {
            lw(aux_gpr(0), gpr_ptr);
            vmv_v_x(src_vec(vec_idx), aux_gpr(0));
        } else {
            vle32_v(src_vec(vec_idx), gpr_ptr);
            add(gpr_ptr, gpr_ptr, reg_bvlen);
        }
        break;
    }
    case ov::element::i8:
    case ov::element::u8: {
        if (broadcast) {
            lb(aux_gpr(0), gpr_ptr);
            vmv_v_x(aux_vec(), aux_gpr(0));
        } else {
            vle8_v(aux_vec(), gpr_ptr);
            add(gpr_ptr, gpr_ptr, reg_bvlen);
        }

        update_vlen(gpr_work_amount, exec_sew, exec_lmul);
        if (src_prc.is_signed())
            vsext_vf4(src_vec(vec_idx), aux_vec());
        else
            vzext_vf4(src_vec(vec_idx), aux_vec());
        break;
    }
    default: {
        OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
    }
    }

    if (one_of(dst_prc, ov::element::f32) && one_of(src_prc, ov::element::i8, ov::element::u8, ov::element::i32))
        vfcvt_f_x_v(src_vec(vec_idx), src_vec(vec_idx)); // int32 -> fp32

    if (one_of(dst_prc, ov::element::i32) && one_of(src_prc, ov::element::f16, ov::element::f32))
        vfcvt_rtz_x_f_v(src_vec(vec_idx), src_vec(vec_idx)); // fp32 -> int32 (round-toward-zero)
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::store_vector(const Xbyak_riscv::Reg& gpr_work_amount, const ov::element::Type& src_prc, const ov::element::Type& dst_prc) {
    OPENVINO_ASSERT(src_prc.size() == sew2bytes(exec_sew), "Incompatible execution SEW and src SEW");
    OPENVINO_ASSERT(one_of(src_prc, ov::element::f32, ov::element::i32), "Unsupported src prc");

    if (one_of(src_prc, ov::element::f32) && one_of(dst_prc, ov::element::i8, ov::element::u8, ov::element::i32))
        vfcvt_rtz_x_f_v(dst_vec(), dst_vec()); // fp32 -> int32 (round-toward-zero)

    if (one_of(src_prc, ov::element::i32) && one_of(dst_prc, ov::element::f16, ov::element::f32))
        vfcvt_f_x_v(dst_vec(), dst_vec()); // int32 -> fp32

    switch (dst_prc) {
    case ov::element::f32:
    case ov::element::i32: {
        vse32_v(dst_vec(), dst_gpr());
        add(dst_gpr(), dst_gpr(), reg_bvlen);
        break;
    }
    case ov::element::i8:
    case ov::element::u8: {
        auto vnclip = [&](const VReg& dst, const VReg& src, float lmul_c, size_t sew) {
            const auto needed_lmul = float2lmul(lmul_c * lmul2float(exec_lmul));
            const auto needed_sew = bytes2sew(sew);
            update_vlen(gpr_work_amount, needed_sew, needed_lmul);
            if (src_prc.is_signed())
                vnclip_wi(dst, src, 0);
            else
                vnclipu_wi(dst, src, 0);
        };
        vnclip(aux_vec(), dst_vec(), 0.5f, 2);
        vnclip(dst_vec(), aux_vec(), 0.25f, 1);

        vse8_v(dst_vec(), dst_gpr());
        add(dst_gpr(), dst_gpr(), reg_bvlen);
        break;
    }
    default: {
        OPENVINO_THROW("dst " + dst_prc.to_string() + " is not supported, src is " + src_prc.to_string());
    }
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
Xbyak_riscv::LMUL jit_uni_eltwise_generic<isa>::compute_exec_lmul(const ov::element::Type& exec_prc) const {
    // TODO: Support any LMUL values
    if (!eltwise_emitter->is_lmul_supported())
        return LMUL::m1;
    for (const auto& post_op_emitter : post_op_emitters) {
        if (!post_op_emitter->is_lmul_supported())
            return LMUL::m1;
    }
    return get_max_lmul(exec_prc);
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
Xbyak_riscv::LMUL jit_uni_eltwise_generic<isa>::get_max_lmul(const ov::element::Type& exec_prc) const {
    OPENVINO_ASSERT(eltwise_emitter, "Eltwise emitter is missed");

    const auto input_count = jep_.inputs_number;
    const auto output_count = 1;
    size_t max_aux_vec_count = eltwise_emitter->aux_vecs_count();
    for (const auto& post_op_emitter : post_op_emitters) {
        max_aux_vec_count = std::max(max_aux_vec_count, post_op_emitter->aux_vecs_count());
    }

    // aux vec is needed for intermediate conversion in load/store
    if (jep_.src_prc[input_count - 1].size() < exec_prc.size() || jep_.dst_prc.size() < exec_prc.size())
        max_aux_vec_count = std::max(max_aux_vec_count, 1lu);

    const auto needed_vec_count = input_count + output_count + max_aux_vec_count + 1; // 1 - mask vec register
    const auto mul = static_cast<size_t>(vec_count / needed_vec_count);
    OPENVINO_ASSERT(mul != 0, "Incorrect configuration!");
    if (mul < 2) return LMUL::m1;
    if (mul < 4) return LMUL::m2;
    if (mul < 8) return LMUL::m4;
    return LMUL::m8;
}

namespace {
struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    ov::intel_cpu::riscv64::jit_generator* host;
    ov::intel_cpu::riscv64::cpu_isa_t host_isa;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template <typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
    }
};

template <>
struct EltwiseEmitter<jit_clamp_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_clamp_emitter>(ctx.host, ctx.host_isa, ctx.opData.alpha, ctx.opData.beta, ctx.exec_prc);
    }
};

template <>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host, ctx.host_isa, ctx.opData.alpha, ctx.opData.beta, ctx.opData.gamma, ctx.exec_prc);
    }
};

template <>
struct EltwiseEmitter<jit_relu_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_relu_emitter>(ctx.host, ctx.host_isa, ctx.opData.alpha, ctx.exec_prc);
    }
};
} // namespace

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
std::shared_ptr<jit_emitter> jit_uni_eltwise_generic<isa>::create_eltwise_emitter(const EltwiseData& data,
                                                                                  const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {nullptr, this, isa, data, exec_prec};

    OV_SWITCH(
        intel_cpu,
        EltwiseEmitter,
        ctx,
        data.algo,
        OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
        OV_CASE(Algorithm::EltwiseClamp, jit_clamp_emitter),
        OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
        OV_CASE(Algorithm::EltwiseExp, jit_exp_emitter),
        OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
        OV_CASE(Algorithm::EltwiseRelu, jit_relu_emitter),
        OV_CASE(Algorithm::EltwiseSigmoid, jit_sigmoid_emitter),
        OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));

    if (!ctx.emitter) {
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");
    }

    return ctx.emitter;
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::compute_eltwise_op() const {
    OPENVINO_ASSERT(eltwise_emitter, "Eltwise emitter is missed for code emission!");

    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_num(); i++) {
        in_idxs.push_back(src_vec(i).getIdx());
    }

    std::vector<size_t> out_idxs;
    out_idxs.push_back(dst_vec().getIdx());

    const auto max_aux_fp_gpr_available = get_max_aux_fp_gpr_count();
    const auto max_aux_gpr_available = get_max_aux_gpr_count();
    const auto max_aux_vec_available = get_max_aux_vec_count();

    std::vector<size_t> aux_vec_idxs;
    for (size_t i = 0; i < std::min(eltwise_emitter->aux_vecs_count(), max_aux_vec_available); i++) {
        aux_vec_idxs.push_back(aux_vec(i).getIdx());
    }

    std::vector<size_t> aux_gpr_idxs;
    for (size_t i = 0; i < std::min(eltwise_emitter->aux_gprs_count(), max_aux_gpr_available); i++) {
        aux_gpr_idxs.push_back(aux_gpr(i).getIdx());
    }

    std::vector<size_t> aux_fp_gpr_idxs;
    for (size_t i = 0; i < std::min(eltwise_emitter->aux_fp_gprs_count(), max_aux_fp_gpr_available); i++) {
        aux_fp_gpr_idxs.push_back(aux_fp_gpr(i).getIdx());
    }

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_vec_idxs, aux_gpr_idxs, aux_fp_gpr_idxs);
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::apply_post_ops() const {
    int input_idx = eltwise_emitter->get_inputs_num();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < eltwise_data_.size(); i++) {
        OPENVINO_ASSERT(post_op_emitters[eltwise_post_op_idx], "Post-op emitter is missed for code emission!");
        std::vector<size_t> in_idxs;
        in_idxs.push_back(dst_vec().getIdx());
        for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++) {
            in_idxs.push_back(src_vec(input_idx++).getIdx());
        }

        std::vector<size_t> out_idxs;
        out_idxs.push_back(dst_vec().getIdx());

        const auto max_aux_fp_gpr_available = get_max_aux_fp_gpr_count();
        const auto max_aux_gpr_available = get_max_aux_gpr_count();
        const auto max_aux_vec_available = get_max_aux_vec_count();

        std::vector<size_t> aux_vec_idxs;
        for (size_t i = 0; i < std::min(eltwise_emitter->aux_vecs_count(), max_aux_vec_available); i++) {
            aux_vec_idxs.push_back(aux_vec(i).getIdx());
        }
  
        std::vector<size_t> aux_gpr_idxs;
        for (size_t i = 0; i < std::min(eltwise_emitter->aux_gprs_count(), max_aux_gpr_available); i++) {
            aux_gpr_idxs.push_back(aux_gpr(i).getIdx());
        }

        std::vector<size_t> aux_fp_gpr_idxs;
        for (size_t i = 0; i < std::min(eltwise_emitter->aux_fp_gprs_count(), max_aux_fp_gpr_available); i++) {
            aux_fp_gpr_idxs.push_back(aux_fp_gpr(i).getIdx());
        }

        post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_vec_idxs, aux_gpr_idxs, aux_fp_gpr_idxs);

        eltwise_post_op_idx++;
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::emit_data() const {
    OPENVINO_ASSERT(eltwise_emitter, "Eltwise emitter is missed for data emission!");
    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        OPENVINO_ASSERT(post_op_emitters[i], "Post-op emitter is missed for data emission!");
        post_op_emitters[i]->emit_data();
    }
}

template struct jit_uni_eltwise_generic<cpu_isa_t::gv>;

}  // namespace riscv64

namespace {
template <typename T>
struct SupportedPrecisions {
    void operator()(std::set<std::vector<element::Type>>& precisions) {
        precisions = T::get_supported_precisions();
    }
};
}  // namespace

using namespace riscv64;

std::set<std::vector<element::Type>> eltwise_precision_helper::get_supported_precisions(const Algorithm& algo) {
    std::set<std::vector<element::Type>> precisions;

    OV_SWITCH(intel_cpu,
              SupportedPrecisions,
              precisions,
              algo,
              OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
              OV_CASE(Algorithm::EltwiseClamp, jit_clamp_emitter),
              OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
              OV_CASE(Algorithm::EltwiseExp, jit_exp_emitter),
              OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
              OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
              OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
              OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
              OV_CASE(Algorithm::EltwiseRelu, jit_relu_emitter),
              OV_CASE(Algorithm::EltwiseSigmoid, jit_sigmoid_emitter),
              OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter));

    if (precisions.empty()) {
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");
    }

    return precisions;
}

}  // ov::intel_cpu
