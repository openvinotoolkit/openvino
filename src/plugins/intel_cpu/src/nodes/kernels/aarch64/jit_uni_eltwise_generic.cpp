// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_uni_eltwise_generic.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;

void jit_uni_eltwise_kernel::operator()(
    const node::jit_eltwise_call_args_ptrs* const_args,
    const jit_eltwise_call_args_indexes* indexes) {
    assert(ker_);
    ker_(const_args, indexes);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
jit_uni_eltwise_generic<isa>::jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                                      const std::vector<EltwiseData>& eltwise_data,
                                                      const std::vector<ov::intel_cpu::Type>& ops_list,
                                                      const dnnl::post_ops& post_ops) :
                                                      jit_uni_eltwise_kernel(jep),
                                                      jit_generator(),
                                                      eltwise_data_(eltwise_data),
                                                      ops_list_(ops_list),
                                                      post_ops_(post_ops) {}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::generate() {
    preamble();

    auto const exec_prc = eltwise_precision_helper::get_precision(jep_.inputs_number, jep_.src_prc, eltwise_data_);

    eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
    for (size_t i = 1; i < eltwise_data_.size(); ++i) {
        post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
    }

    const auto &jep = jep_;

    XReg param2 = abi_param2;
    const int offset_count = jep.input_size - 1;

    // ptrs initializing
    if (jep.use_runtime_ptrs) {
        for (size_t i = 0; i < jep.inputs_number; i++) {
            ldr(start_to_offsets, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_offsets) + i * sizeof(size_t))));
            ldr(get_src_reg(i), ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr[0]) + i * sizeof(size_t))));
            XReg offset_reg = get_aux_gpr(0); // X_TMP_0;
            XReg index_reg = get_aux_gpr(1);  // X_TMP_1;
            for (int j = 0; j < offset_count; j++) {
                ldr(offset_reg, ptr(start_to_offsets, static_cast<int32_t>(j * sizeof(size_t))));
                ldr(index_reg, ptr(reg_indexes, static_cast<int32_t>(j * sizeof(size_t))));
                madd(get_src_reg(i), offset_reg, index_reg, get_src_reg(i));
            }
        }

        ldr(start_to_offsets, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_offsets))));
        ldr(reg_dst, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr))));
        XReg offset_reg = get_aux_gpr(0); // X_TMP_0;
        XReg index_reg = get_aux_gpr(1);  // X_TMP_1;
        for (int j = 0; j < offset_count; j++) {
            ldr(offset_reg, ptr(start_to_offsets, static_cast<int32_t>(j * sizeof(size_t))));
            ldr(index_reg, ptr(reg_indexes, static_cast<int32_t>(j * sizeof(size_t))));
            madd(reg_dst, offset_reg, index_reg, reg_dst);
        }

        mov(reg_oc_off, 0);

        ldr(reg_work_amount, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, work_amount))));
    } else {
        auto init_ptrs_with_offsets = [this, offset_count, param2](XReg pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    XReg offset_reg(get_aux_gpr(0));
                    mov(offset_reg, offsets[j]);

                    XReg index_reg(get_aux_gpr(1));
                    ldr(index_reg, ptr(param2, static_cast<int32_t>(j * sizeof(size_t))));
                    madd(pointer, offset_reg, index_reg, pointer);
                }
            }
        };

        for (size_t i = 0; i < jep.inputs_number; i++) {
            ldr(get_src_reg(i), ptr(param1, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, src_ptr) + i * sizeof(size_t))));
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
        }

        ldr(reg_dst, ptr(reg_const_params, static_cast<int32_t>(offsetof(node::jit_eltwise_call_args_ptrs, dst_ptr))));
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        mov(reg_oc_off, 0);
        init_ptrs_with_offsets(reg_oc_off, jep.oc_offsets);

        mov(reg_work_amount, jep.work_amount);
    }

    Label unroll_loop_label;
    Label unroll_loop_end_label;
    Label main_loop_label;
    Label main_loop_end_label;
    Label tail_loop_label;
    Label tail_loop_end_label;

    for (size_t i = 0; i < jep.inputs_number; i++) {
        if (jep.src_size[i] == 1) {
            load_vector(get_vmm_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, true);
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

        L(unroll_loop_label);
        {
            const size_t loop_step = min_src_size;
            const size_t vec_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

            cmp(reg_work_amount, loop_step);
            b(LO, unroll_loop_end_label);

            for (size_t j = 0; j < min_src_size / vec_step; j++) {
                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        load_vector(get_vmm_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, false, j * vec_step * jep.src_prc[i].size());
                    }
                }

                compute_eltwise_op();

                apply_post_ops();

                store_vector(reg_dst, vmm_dst, exec_prc, jep.dst_prc, j * vec_step * jep.dst_prc.size());
            }

            size_t tail_start = min_src_size - min_src_size % vec_step;
            for (size_t j = tail_start; j < min_src_size; j++) {
                for (size_t i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1) {
                        load_scalar(get_scl_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, j * jep.src_prc[i].size());
                    }
                }

                compute_eltwise_op();

                apply_post_ops();

                SReg sc_dst_reg{vmm_dst.getIdx()};
                store_scalar(reg_dst, sc_dst_reg, exec_prc, jep.dst_prc, j * jep.dst_prc.size());
            }

            for (size_t i = 0; i < jep.inputs_number; i++)
                if (jep.src_size[i] == jep.dst_size)
                    add(get_src_reg(i), get_src_reg(i), jep.src_prc[i].size() * loop_step);

            add(reg_dst, reg_dst, jep.dst_prc.size() * loop_step);
            sub(reg_work_amount, reg_work_amount, loop_step);
            if (jep_.oc_size > 1 && jep_.oc_size != min_src_size)
                add(reg_oc_off, reg_oc_off, loop_step * sizeof(float));

            b(AL, unroll_loop_label);
        }

        L(unroll_loop_end_label);
    }

    if (min_src_size == jep.dst_size) {
        L(main_loop_label);
        {
            const size_t vlen = cpu_isa_traits<isa>::vlen;
            const size_t exec_prc_size = exec_prc.size();
            const size_t loop_step = vlen / exec_prc_size;

            cmp(reg_work_amount, loop_step);
            b(LO, main_loop_end_label);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    load_vector(get_vmm_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc, false);
                }
            }

            compute_eltwise_op();

            apply_post_ops();

            store_vector(reg_dst, vmm_dst, exec_prc, jep.dst_prc);

            for (size_t i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1) {
                    add(get_src_reg(i), get_src_reg(i), jep.src_prc[i].size() * loop_step);
                }
            }

            add(reg_dst, reg_dst, jep.dst_prc.size() * loop_step);
            sub(reg_work_amount, reg_work_amount, loop_step);
            if (jep_.oc_size > 1)
                add(reg_oc_off, reg_oc_off, loop_step * sizeof(float));

            b(AL, main_loop_label);
        }
        L(main_loop_end_label);
    }

    L(tail_loop_label);
    {
        const size_t loop_step = 1;

        cmp(reg_work_amount, 0x0);
        b(EQ, tail_loop_end_label);

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1) {
                load_scalar(get_scl_reg(i), get_src_reg(i), jep.src_prc[i], exec_prc);
            }
        }

        compute_eltwise_op();

        apply_post_ops();

        SReg sc_dst_reg{vmm_dst.getIdx()};
        store_scalar(reg_dst, sc_dst_reg, exec_prc, jep.dst_prc);

        for (size_t i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1) {
                add(get_src_reg(i), get_src_reg(i), jep.src_prc[i].size() * loop_step);
            }
        }

        add(reg_dst, reg_dst, jep.dst_prc.size() * loop_step);
        sub(reg_work_amount, reg_work_amount, loop_step);
        if (jep_.oc_size > 1)
            add(reg_oc_off, reg_oc_off, loop_step * sizeof(float));

        b(AL, tail_loop_label);
    }
    L(tail_loop_end_label);

    postamble();

    eltwise_emitter->emit_data();
    for (size_t i = 0; i < post_op_emitters.size(); i++) {
        post_op_emitters[i]->emit_data();
    }
}

namespace utils {
template <typename T1, typename T2>
void load_vector(const T1& data_lane,
                 const T2& data_lanes,
                 const Xbyak_aarch64::XReg &ptr_reg,
                 const int64_t offset,
                 const bool broadcast,
                 jit_generator* h) {
    if (broadcast) {
        if (offset == 0) {
            h->ld1r(data_lane, ptr(ptr_reg));
        } else {
            h->add_imm(h->X_DEFAULT_ADDR, ptr_reg, offset, h->X_TMP_0);
            h->ld1r(data_lane, ptr(h->X_DEFAULT_ADDR));
        }
    } else {
        if (offset == 0) {
            h->ld1(data_lanes, ptr(ptr_reg));
        } else {
            h->add_imm(h->X_DEFAULT_ADDR, ptr_reg, offset, h->X_TMP_0);
            h->ld1(data_lanes, ptr(h->X_DEFAULT_ADDR));
        }
    }
}
} // namespace utils

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::load_vector(const TReg& data,
                                               const XReg& ptr_reg,
                                               const ov::element::Type& src_prc,
                                               const ov::element::Type& dst_prc,
                                               const bool broadcast,
                                               const int32_t ptr_offset) {
    switch (src_prc) {
        case ov::element::f16: {
            utils::load_vector(data.h, data.h4, ptr_reg, ptr_offset, broadcast, this);
            break;
        }
        case ov::element::f32:
        case ov::element::i32: {
            if (broadcast) {
                jit_generator::uni_ld1rw(data.s, ptr_reg, ptr_offset);
            } else {
                jit_generator::uni_ldr(data, ptr_reg, ptr_offset);
            }
            break;
        }
        case ov::element::i8: {
            utils::load_vector(data.b, data.s, ptr_reg, ptr_offset, broadcast, this);
            sshll(data.h8, data.b8, 0);
            sshll(data.s4, data.h4, 0);
            break;
        }
        case ov::element::u8: {
            utils::load_vector(data.b, data.s, ptr_reg, ptr_offset, broadcast, this);
            ushll(data.h8, data.b8, 0);
            ushll(data.s4, data.h4, 0);
            break;
        }
        default: {
            OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
        }
    }

    if (dst_prc != src_prc) {
        switch (dst_prc) {
            case ov::element::f32:
                switch (src_prc) {
                    case ov::element::f16: {
                        fcvtl(data.s4, data.h4);
                        break;
                    }
                    case ov::element::i32: {
                        scvtf(data.s, data.s);
                        break;
                    }
                    case ov::element::i8: {
                        scvtf(data.s, data.s);
                        break;
                    }
                    case ov::element::u8: {
                        ucvtf(data.s, data.s);
                        break;
                    }
                    default:
                        OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
                }
                break;
            default:
                OPENVINO_THROW("dst_prc " + dst_prc.to_string() + " is not supported, src_prc is " + src_prc.to_string());
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::load_scalar(const SReg& data,
                                               const XReg& ptr,
                                               const ov::element::Type& src_prc,
                                               const ov::element::Type& dst_prc,
                                               const int32_t ptr_offset) {
    switch (src_prc) {
        case ov::element::f16: {
            ldr(Xbyak_aarch64::HReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::f32:
        case ov::element::i32: {
            ldr(data, Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::i8: {
            ldr(Xbyak_aarch64::BReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));

            // scalar is loaded, operates with vector
            TReg vec(data.getIdx());
            sshll(vec.h8, vec.b8, 0);
            sshll(vec.s4, vec.h4, 0);
            break;
        }
        case ov::element::u8: {
            ldr(Xbyak_aarch64::BReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));

            // scalar is loaded, operates with vector
            TReg vec(data.getIdx());
            ushll(vec.h8, vec.b8, 0);
            ushll(vec.s4, vec.h4, 0);
            break;
        }
        default: {
            OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
        }
    }

    if (dst_prc != src_prc) {
        switch (dst_prc) {
            case ov::element::f32:
                switch (src_prc) {
                    case ov::element::f16: {
                        fcvt(Xbyak_aarch64::SReg(data.getIdx()), Xbyak_aarch64::HReg(data.getIdx()));
                        break;
                    }
                    case ov::element::i32:
                    case ov::element::i8: {
                        scvtf(Xbyak_aarch64::SReg(data.getIdx()), Xbyak_aarch64::SReg(data.getIdx()));
                        break;
                    }
                    case ov::element::u8: {
                        ucvtf(Xbyak_aarch64::SReg(data.getIdx()), Xbyak_aarch64::SReg(data.getIdx()));
                        break;
                    }
                    default:
                        OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
                }
                break;
            default:
                OPENVINO_THROW("dst_prc " + dst_prc.to_string() + " is not supported, src_prc is " + src_prc.to_string());
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::store_vector(const XReg& ptr,
                                                const TReg& data,
                                                const ov::element::Type& src_prc,
                                                const ov::element::Type& dst_prc,
                                                const int32_t ptr_offset) {
    if (src_prc != dst_prc) {
        switch (src_prc) {
            case ov::element::f32: {
                switch (dst_prc) {
                    case ov::element::f16: {
                        fcvtn(data.h4, data.s4);
                        break;
                    }
                    case ov::element::i32: {
                        fcvtns(data.s, data.s);
                        break;
                    }
                    case ov::element::i8: {
                        fcvtms(data.s, data.s);
                        xtn(data.h4, data.s4);
                        xtn(data.b8, data.h8);
                        break;
                    }
                    case ov::element::u8: {
                        fcvtmu(data.s, data.s);
                        xtn(data.h4, data.s4);
                        xtn(data.b8, data.h8);
                        break;
                    }
                    default: {
                        OPENVINO_THROW("dst_prc " + dst_prc.to_string() + " is not supported, src_prc is " + src_prc.to_string());
                    }
                }
                break;
            }
            default: {
                OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
            }
        }
    }

    switch (dst_prc) {
        case ov::element::f16: {
            str(Xbyak_aarch64::DReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::f32:
        case ov::element::i32: {
            str(Xbyak_aarch64::QReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::i8:
        case ov::element::u8: {
            str(Xbyak_aarch64::SReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        default: {
            OPENVINO_THROW("dst_prc " + dst_prc.to_string() + " is not supported, src_ptr is " + src_prc.to_string());
        }
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::store_scalar(const XReg& ptr,
                                                const SReg& data,
                                                const ov::element::Type& src_prc,
                                                const ov::element::Type& dst_prc,
                                                const int32_t ptr_offset) {
    if (src_prc != dst_prc) {
        switch (src_prc) {
            case ov::element::f32: {
                switch (dst_prc) {
                    case ov::element::f16: {
                        fcvt(Xbyak_aarch64::HReg(data.getIdx()), data);
                        break;
                    }
                    case ov::element::i32: {
                        fcvtns(data, data);
                        break;
                    }
                    case ov::element::i8: {
                        TReg vec_data(data.getIdx());
                        fcvtms(vec_data.s, vec_data.s);
                        xtn(vec_data.h4, vec_data.s4);
                        xtn(vec_data.b8, vec_data.h8);
                        break;
                    }
                    case ov::element::u8: {
                        TReg vec_data(data.getIdx());
                        fcvtmu(vec_data.s, vec_data.s);
                        xtn(vec_data.h4, vec_data.s4);
                        xtn(vec_data.b8, vec_data.h8);
                        break;
                    }
                    default: {
                        OPENVINO_THROW("dst_prc " + dst_prc.to_string() + " is not supported, src_prc is " + src_prc.to_string());
                    }
                }
                break;
            }
            default: {
                OPENVINO_THROW("src_prc " + src_prc.to_string() + " is not supported, dst_prc is " + dst_prc.to_string());
            }
        }
    }

    switch (dst_prc) {
        case ov::element::f16: {
            str(Xbyak_aarch64::HReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::i32:
        case ov::element::f32: {
            str(data, Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        case ov::element::i8:
        case ov::element::u8: {
            str(Xbyak_aarch64::BReg(data.getIdx()), Xbyak_aarch64::ptr(ptr, ptr_offset));
            break;
        }
        default: {
            OPENVINO_THROW("dst_prc " + src_prc.to_string() + " is not supported, src_prc is " + src_prc.to_string());
        }
    }
}

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    dnnl::impl::cpu::aarch64::jit_generator *host;
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa;
    const EltwiseData& opData;
    ov::element::Type exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_swish_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_swish_emitter>(ctx.host,
                                                          ctx.host_isa,
                                                          ctx.opData.alpha,
                                                          ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_elu_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_elu_emitter>(ctx.host,
                                                          ctx.host_isa,
                                                          ctx.opData.alpha,
                                                          ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_clamp_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_clamp_emitter>(ctx.host,
                                                          ctx.host_isa,
                                                          ctx.opData.alpha,
                                                          ctx.opData.beta,
                                                          ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host,
                                                                 ctx.host_isa,
                                                                 ctx.opData.alpha,
                                                                 ctx.opData.beta,
                                                                 ctx.opData.gamma,
                                                                 ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_is_inf_emitter> {
    void operator()(EltwiseEmitterContext& ctx) {
        ctx.emitter = std::make_shared<jit_is_inf_emitter>(ctx.host,
                                                           ctx.host_isa,
                                                           ctx.opData.alpha,
                                                           ctx.opData.beta,
                                                           ctx.exec_prc);
    }
};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
std::shared_ptr<jit_emitter> jit_uni_eltwise_generic<isa>::create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec) {
    EltwiseEmitterContext ctx = {
        nullptr,
        this,
        isa,
        data,
        exec_prec
    };

    OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, data.algo,
    OV_CASE(Algorithm::EltwiseAbs, ov::intel_cpu::aarch64::jit_abs_emitter),
    OV_CASE(Algorithm::EltwiseAdd, ov::intel_cpu::aarch64::jit_add_emitter),
    OV_CASE(Algorithm::EltwiseClamp, ov::intel_cpu::aarch64::jit_clamp_emitter),
    OV_CASE(Algorithm::EltwiseDivide, ov::intel_cpu::aarch64::jit_divide_emitter),
    OV_CASE(Algorithm::EltwiseElu, ov::intel_cpu::aarch64::jit_elu_emitter),
    OV_CASE(Algorithm::EltwiseEqual, ov::intel_cpu::aarch64::jit_equal_emitter),
    OV_CASE(Algorithm::EltwiseExp, ov::intel_cpu::aarch64::jit_exp_emitter),
    OV_CASE(Algorithm::EltwiseFloor, ov::intel_cpu::aarch64::jit_floor_emitter),
    OV_CASE(Algorithm::EltwiseHswish, ov::intel_cpu::aarch64::jit_hswish_emitter),
    OV_CASE(Algorithm::EltwiseIsFinite, ov::intel_cpu::aarch64::jit_is_finite_emitter),
    OV_CASE(Algorithm::EltwiseIsInf, ov::intel_cpu::aarch64::jit_is_inf_emitter),
    OV_CASE(Algorithm::EltwiseLessEqual, ov::intel_cpu::aarch64::jit_less_equal_emitter),
    OV_CASE(Algorithm::EltwiseLogicalNot, ov::intel_cpu::aarch64::jit_logical_not_emitter),
    OV_CASE(Algorithm::EltwiseLogicalXor, ov::intel_cpu::aarch64::jit_logical_xor_emitter),
    OV_CASE(Algorithm::EltwiseIsNaN, ov::intel_cpu::aarch64::jit_is_nan_emitter),
    OV_CASE(Algorithm::EltwiseMaximum, ov::intel_cpu::aarch64::jit_maximum_emitter),
    OV_CASE(Algorithm::EltwiseMinimum, ov::intel_cpu::aarch64::jit_minimum_emitter),
    OV_CASE(Algorithm::EltwiseMish, ov::intel_cpu::aarch64::jit_mish_emitter),
    OV_CASE(Algorithm::EltwiseGeluErf, ov::intel_cpu::aarch64::jit_gelu_erf_emitter),
    OV_CASE(Algorithm::EltwiseGeluTanh, ov::intel_cpu::aarch64::jit_gelu_tanh_emitter),
    OV_CASE(Algorithm::EltwiseGreater, ov::intel_cpu::aarch64::jit_greater_emitter),
    OV_CASE(Algorithm::EltwiseGreaterEqual, ov::intel_cpu::aarch64::jit_greater_equal_emitter),
    OV_CASE(Algorithm::EltwiseMulAdd, ov::intel_cpu::aarch64::jit_mul_add_emitter),
    OV_CASE(Algorithm::EltwiseMod, ov::intel_cpu::aarch64::jit_mod_emitter),
    OV_CASE(Algorithm::EltwiseMultiply, ov::intel_cpu::aarch64::jit_multiply_emitter),
    OV_CASE(Algorithm::EltwisePowerStatic, ov::intel_cpu::aarch64::jit_power_static_emitter),
    OV_CASE(Algorithm::EltwisePrelu, ov::intel_cpu::aarch64::jit_prelu_emitter),
    OV_CASE(Algorithm::EltwiseRelu, ov::intel_cpu::aarch64::jit_relu_emitter),
    OV_CASE(Algorithm::EltwiseSelect, ov::intel_cpu::aarch64::jit_select_emitter),
    OV_CASE(Algorithm::EltwiseSigmoid, ov::intel_cpu::aarch64::jit_sigmoid_emitter),
    OV_CASE(Algorithm::EltwiseSoftSign, ov::intel_cpu::aarch64::jit_soft_sign_emitter),
    OV_CASE(Algorithm::EltwiseSqrt, ov::intel_cpu::aarch64::jit_sqrt_emitter),
    OV_CASE(Algorithm::EltwiseSubtract, ov::intel_cpu::aarch64::jit_subtract_emitter),
    OV_CASE(Algorithm::EltwiseSwish, ov::intel_cpu::aarch64::jit_swish_emitter),
    OV_CASE(Algorithm::EltwiseTanh, ov::intel_cpu::aarch64::jit_tanh_emitter));

    if (!ctx.emitter)
        OPENVINO_THROW("Unsupported operation type '" + algToString(data.algo) + "' for Eltwise emitter");

    return ctx.emitter;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::compute_eltwise_op() {
    std::vector<size_t> in_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_inputs_count(); i++) {
        in_idxs.push_back(get_vmm_reg(i).getIdx());
    }

    std::vector<size_t> aux_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_vecs_count(); i++) {
        aux_idxs.push_back(get_aux_vmm(i).getIdx());
    }

    std::vector<size_t> out_idxs;
    out_idxs.push_back(vmm_dst.getIdx());

    std::vector<size_t> gpr_idxs;
    for (size_t i = 0; i < eltwise_emitter->get_aux_gprs_count(); i++) {
        gpr_idxs.push_back(get_aux_gpr(i).getIdx());
    }

    eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs, gpr_idxs);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_uni_eltwise_generic<isa>::apply_post_ops() {
    int input_idx = eltwise_emitter->get_inputs_count();
    int eltwise_post_op_idx = 0;
    for (size_t i = 1; i < ops_list_.size(); i++) {
        if (ops_list_[i] == ov::intel_cpu::Type::Eltwise) {
            std::vector<size_t> in_idxs;
            in_idxs.push_back(vmm_dst.getIdx());
            for (size_t j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_count(); j++)
                in_idxs.push_back(get_vmm_reg(input_idx++).getIdx());

            std::vector<size_t> out_idxs;
            out_idxs.push_back(vmm_dst.getIdx());

            std::vector<size_t> aux_vmm_idxs;
            for (size_t j = 0; j < post_op_emitters[eltwise_post_op_idx]->get_aux_vecs_count(); j++)
                aux_vmm_idxs.push_back(get_aux_vmm(j).getIdx());

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

namespace {

template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<std::vector<element::Type>> &precisions) {
        precisions = T::get_supported_precisions();
    }
};

static void set_intersection(const std::set<std::vector<element::Type>>& precisions1,
                      const std::set<std::vector<element::Type>>& precisions2,
                      std::set<std::vector<element::Type>>& intersection) {
    std::map<element::Type, size_t> intersection_types;

    for (auto it1 = precisions1.begin(); it1 != precisions1.end(); ++it1) {
        for (auto it2 = precisions2.begin(); it2 != precisions2.end(); ++it2) {
            const auto& it1_precisions = *it1;
            // all element types are equal
            if (it1_precisions[0] == (*it2)[0]) {
                // first precisions size is used
                intersection_types.emplace(it1_precisions[0], it1_precisions.size());
            }
        }
    }

    for (auto it = intersection_types.begin(); it != intersection_types.end(); ++it) {
        intersection.insert(std::vector<element::Type>(it->second, it->first));
    }
}
} // namespace

ov::element::Type eltwise_precision_helper::get_precision(const size_t inputs_number,
                                                          const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                                          const std::vector<EltwiseData>& eltwise_data) {
    ov::element::Type exec_prc = ov::element::undefined;

    const auto algorithm = eltwise_data.front().algo;
    std::set<std::vector<element::Type>> supported_precision_intersection = get_supported_precisions(algorithm);

    for (size_t i = 1; i < eltwise_data.size(); ++i) {
        std::set<std::vector<element::Type>> prcs = get_supported_precisions(eltwise_data[i].algo);
        std::set<std::vector<element::Type>> prcs_intersect = {};

        set_intersection(supported_precision_intersection, prcs, prcs_intersect);

        supported_precision_intersection = prcs_intersect;
    }

    static const element::Type exec_precisions_priority[] = {
        element::f16,
        element::f32
    };

    for (const auto prc : exec_precisions_priority) {
        if (std::any_of(
            supported_precision_intersection.begin(),
            supported_precision_intersection.end(),
            [&prc](const std::vector<element::Type>& precisions) { return std::find(precisions.begin(), precisions.end(), prc) != precisions.end(); })) {
            exec_prc = prc;
            break;
        }
    }

    for (size_t i = 0; i < inputs_number; i++) {
        if (src_prc[i] != exec_prc) {
            exec_prc = ov::element::f32;
            break;
        }
    }

    if (exec_prc == ov::element::undefined) {
        OPENVINO_THROW("Eltwise jitter failed to specify execution precision for Eltwise node");
    }

    return exec_prc;
}

std::set<std::vector<element::Type>> eltwise_precision_helper::get_supported_precisions(const Algorithm& algo) {
    std::set<std::vector<element::Type>> precisions;

    OV_SWITCH(intel_cpu, SupportedPrecisions, precisions, algo,
        OV_CASE(Algorithm::EltwiseRelu, jit_relu_emitter),
        OV_CASE(Algorithm::EltwiseAbs, jit_abs_emitter),
        OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
        OV_CASE(Algorithm::EltwiseClamp, jit_clamp_emitter),
        OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
        OV_CASE(Algorithm::EltwiseElu, jit_elu_emitter),
        OV_CASE(Algorithm::EltwiseEqual, jit_equal_emitter),
        OV_CASE(Algorithm::EltwiseExp, jit_exp_emitter),
        OV_CASE(Algorithm::EltwiseFloor, jit_floor_emitter),
        OV_CASE(Algorithm::EltwiseGeluErf, jit_gelu_erf_emitter),
        OV_CASE(Algorithm::EltwiseGeluTanh, jit_gelu_tanh_emitter),
        OV_CASE(Algorithm::EltwiseGreater, jit_greater_emitter),
        OV_CASE(Algorithm::EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Algorithm::EltwiseHswish, jit_hswish_emitter),
        OV_CASE(Algorithm::EltwiseIsFinite, jit_is_finite_emitter),
        OV_CASE(Algorithm::EltwiseIsInf, jit_is_inf_emitter),
        OV_CASE(Algorithm::EltwiseIsNaN, jit_is_nan_emitter),
        OV_CASE(Algorithm::EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(Algorithm::EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(Algorithm::EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(Algorithm::EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(Algorithm::EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(Algorithm::EltwiseMish, jit_mish_emitter),
        OV_CASE(Algorithm::EltwiseMod, jit_mod_emitter),
        OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
        OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(Algorithm::EltwiseSelect, jit_select_emitter),
        OV_CASE(Algorithm::EltwiseSigmoid, jit_sigmoid_emitter),
        OV_CASE(Algorithm::EltwiseSoftSign, jit_soft_sign_emitter),
        OV_CASE(Algorithm::EltwiseSqrt, jit_sqrt_emitter),
        OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(Algorithm::EltwiseSwish, jit_swish_emitter),
        OV_CASE(Algorithm::EltwiseTanh, jit_tanh_emitter));
    if (precisions.empty())
        OPENVINO_THROW("Unsupported operation type for Eltwise emitter");

    return precisions;
}

template struct jit_uni_eltwise_generic<cpu_isa_t::asimd>;

}  // namespace aarch64
}  // namespace intel_cpu
}  // namespace ov
