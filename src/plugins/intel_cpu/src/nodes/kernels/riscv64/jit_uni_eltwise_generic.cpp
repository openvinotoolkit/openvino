// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

#include "emitters/plugin/riscv64/jit_eltwise_emitters.hpp"


namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) offsetof(jit_eltwise_call_args_ptrs, field)

jit_uni_eltwise_generic::jit_uni_eltwise_generic(jit_eltwise_params jep, std::vector<EltwiseData> eltwise_data)
    : jit_uni_eltwise_kernel(std::move(jep)),
      jit_generator(),
      eltwise_data_(std::move(eltwise_data)) {}

void jit_uni_eltwise_generic::generate() {
    preamble();

    static const std::vector<element::Type> exec_precisions_priority = { element::f32 };
    auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number,
                                                                  jep_.src_prc,
                                                                  eltwise_data_,
                                                                  exec_precisions_priority);

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    }

    const auto& jep = jep_;
    const int offset_count = jep.input_size - 1;
    const LMUL lmul = LMUL::m1;

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
                    // what's about 64bit?
                    li(reg_tmp_0, static_cast<int>(offsets[j]));
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

        li(reg_work_amount, static_cast<int>(jep.work_amount));
    }

    vsetvli(reg_vlen, reg_work_amount, SEW::e32, lmul);
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            flw(f0, src_gpr(i));
            vfmv_v_f(src_vec(i, lmul), f0);
        }
    }

    size_t min_src_size = jep.dst_size;
    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] != 1)
            min_src_size = std::min(min_src_size, jep.src_size[i]);
    }

    if (min_src_size == jep.dst_size) {
        std::cout << "EQUAL\n";
        Label loop_begin;
        Label loop_end;

        L(loop_begin);
        {
            beqz(reg_work_amount, loop_end);

            vsetvli(reg_vlen, reg_work_amount, SEW::e32, lmul);
            sub(reg_work_amount, reg_work_amount, reg_vlen);
            slli(reg_vlen, reg_vlen, 2); // in bytes

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    vle32_v(src_vec(i, lmul), src_gpr(i));
                    add(src_gpr(i), src_gpr(i), reg_vlen);
                }
            }

            compute_eltwise_op(lmul);
            apply_post_ops(lmul);

            vse32_v(dst_vec(), dst_gpr());
            add(dst_gpr(), dst_gpr(), reg_vlen);

            bnez(reg_work_amount, loop_begin);
        }
        L(loop_end);
    }

    if (min_src_size != jep.dst_size) {
        std::cout << "UNEQUAL\n";
        std::cout << jep.src_size[0] << " " << jep.src_size[1] << jep.dst_size << std::endl;
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

            li(reg_loop_step, min_src_size);
            L(inner_loop_begin);
            {
                vsetvli(reg_vlen, reg_loop_step, SEW::e32, lmul);

                sub(reg_loop_step, reg_loop_step, reg_vlen);
                slli(reg_vlen, reg_vlen, 2);  // to bytes

                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        vle32_v(src_vec(i, lmul), src_aux_gpr(i));
                        add(src_aux_gpr(i), src_aux_gpr(i), reg_vlen);
                    }
                }

                compute_eltwise_op(lmul);
                apply_post_ops(lmul);

                vse32_v(dst_vec(), dst_gpr());
                add(dst_gpr(), dst_gpr(), reg_vlen);

                bnez(reg_loop_step, inner_loop_begin);
            }

            const auto reg_tmp = reg_loop_step;
            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] == jep.dst_size) {
                    li(reg_tmp, jep.src_prc[i].size() * min_src_size);
                    add(src_gpr(i), src_gpr(i), reg_tmp);
                }
            }

            li(reg_loop_step, min_src_size);
            sub(reg_work_amount, reg_work_amount, reg_loop_step);
            j_(loop_begin);
        }

        L(loop_end);
    }

    postamble();
    emit_data();
}

void jit_uni_eltwise_generic::emit_data() const {
    OPENVINO_ASSERT(eltwise_emitter, "Emitter is missed");
    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        post_op_emitters[i]->emit_data();
    }
}

int jit_uni_eltwise_generic::lmul2int(const LMUL lmul) const {
    switch (lmul) {
    case LMUL::m1:
        return 1;
    case LMUL::m2:
        return 2;
    case LMUL::m4:
        return 4;
    case LMUL::m8:
        return 8;
    default: {
        OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(static_cast<uint32_t>(lmul)));
    }
    }
}

namespace {
struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    ov::intel_cpu::riscv64::jit_generator* host;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template <typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.exec_prc);
    }
};
} // namespace

std::shared_ptr<jit_emitter> jit_uni_eltwise_generic::create_eltwise_emitter(const EltwiseData& data,
                                                                             const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {nullptr, this, data, exec_prec};

    OV_SWITCH(
        intel_cpu,
        EltwiseEmitter,
        ctx,
        data.algo,
        OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::riscv64::jit_add_emitter));

    if (!ctx.emitter) {
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");
    }

    return ctx.emitter;
}

void jit_uni_eltwise_generic::compute_eltwise_op(const Xbyak_riscv::LMUL lmul) {
    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_num(); i++) {
        in_idxs.push_back(src_vec(i, lmul).getIdx());
    }

    std::vector<size_t> aux_idxs;
    //for (size_t i = 0; i < eltwise_emitter->aux_vecs_count(); i++) {
    //    aux_idxs.push_back(aux_vmm(i).getIdx());
    //}

    std::vector<size_t> out_idxs;
    out_idxs.push_back(dst_vec().getIdx());

    std::vector<size_t> gpr_idxs;
    //for (size_t i = 0; i < eltwise_emitter->aux_gprs_count(); i++) {
    //    gpr_idxs.push_back(aux_gpr(i).getIdx());
    //}

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs, gpr_idxs);
}

void jit_uni_eltwise_generic::apply_post_ops(const Xbyak_riscv::LMUL lmul) {
    int input_idx = eltwise_emitter->get_inputs_num();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < eltwise_data_.size(); i++) {
        std::vector<size_t> in_idxs;
        in_idxs.push_back(dst_vec().getIdx());
        for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++) {
            in_idxs.push_back(src_vec(input_idx++, lmul).getIdx());
        }

        std::vector<size_t> out_idxs;
        out_idxs.push_back(dst_vec().getIdx());

        std::vector<size_t> aux_vmm_idxs;
        //for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_vecs_count(); j++) {
        //    aux_vmm_idxs.push_back(aux_vmm(j).getIdx());
        //}

        std::vector<size_t> aux_gpr_idxs;
        //for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_gprs_count(); j++) {
        //    aux_gpr_idxs.push_back(aux_gpr(j).getIdx());
        //}

        post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_vmm_idxs, aux_gpr_idxs);

        eltwise_post_op_idx++;
    }
}
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
              OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter));

    if (precisions.empty()) {
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");
    }

    return precisions;
}

}  // namespace intel_cpu
}  // namespace ov
