/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace inner_product_utils {

using namespace dnnl::impl::cpu::inner_product_utils;
using namespace Xbyak;
using namespace data_type;

template <cpu_isa_t isa>
struct jit_pp_kernel_t : public pp_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(inner_product_utils::jit_pp_kernel_t);

    jit_pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt,
            data_type_t acc_dt, const memory_desc_t *dst_md, bool skip_sum);

    void operator()(void *dst, const void *acc, const char *bias,
            const float *scales, size_t start, size_t dst_logical_off,
            size_t dim1_off, size_t end, size_t runtime_oc, dim_t dst_mb_stride,
            const float *dst_zero_points,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            size_t first_mb_matrix_addr_off, const exec_ctx_t &ctx,
            const memory_desc_t &dst_md) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    enum class arg_t { dst, acc, bias, stack, scale, sum };
    enum class data_op_t { load, store };

    void apply_postops(const bool apply_mask, const int vmm_idx,
            const size_t offset, bool runtime_tail_mask);
    void prepare_mask(const size_t tail);
    void load_no_tail(const Vmm &v, Xbyak::Address op, const data_type_t dt);
    void load_tail(const Vmm &v, const arg_t arg_num, const size_t off,
            const data_type_t dt, const size_t tail);
    void load_and_cvt(const Vmm &v, const arg_t arg_num, const size_t off,
            const size_t tail, bool do_cvt = true);
    // convert and store instances for each case of Vmm
    void cvt_and_store(const Xbyak::Zmm &v, const arg_t arg_num,
            const size_t off, const size_t tail);
    void cvt_and_store(const Xbyak::Ymm &v, const arg_t arg_num,
            const size_t off, const size_t tail);
    void cvt_and_store(const Xbyak::Xmm &v, const arg_t arg_num,
            const size_t off, const size_t tail);
    void runtime_tail_load_cvt(const Vmm &v, const arg_t arg_num,
            const size_t off, bool cvt = true);
    void runtime_tail_cvt_store(
            const Vmm &v, const arg_t arg_num, const size_t off);
    void data_copy(const Vmm &v, const arg_t arg_num, const size_t off,
            data_op_t data_op, const size_t tail,
            const bool is_needed_runtime_tail_process,
            const bool do_cvt = true);
    void generate() override;
    void compute_oc_channel_blk();
    void compute_mb_blk(); // vectorize across minibatch

    void advance_binary_postops_off(const size_t offset);

    void advance_binary_postops_off(const Xbyak::Reg64 offset);

    template <typename T>
    void advance_binary_postops_per_oc_off(const T &offset);

    void update_binary_postops_per_tensor_off();

    template <typename T>
    void advance_binary_postops_channel_bcast_off(const T &offset);

    struct ker_args_t {
        char *dst = nullptr;
        const char *acc = nullptr;
        const char *bias = nullptr;
        const float *scales = nullptr;
        const float *dst_zero_points = nullptr;
        float nslope = 0;
        size_t oc = 0;
        size_t len = 0;
        size_t oc_offset = 0;
        size_t dim1_off = 0;
        size_t dst_logical_off = 0;
        size_t first_mb_matrix_addr_off = 0;
        dim_t dst_mb_stride = 0;
        const void *post_ops_binary_rhs_arg_vec = nullptr;
        const void *dst_orig = nullptr;
    };

    const bool is_avx512_ = utils::one_of(isa, avx512_core, avx512_core_bf16);
    static constexpr cpu_isa_t inject_isa_
            = isa == avx512_core_bf16 ? avx512_core : isa;
    std::unique_ptr<injector::jit_uni_postops_injector_t<inject_isa_>>
            postops_injector_;

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

#ifdef _WIN32
    const Xbyak::Reg64 &reg_binary_inj_param_ = abi_not_param1;
#else
    const Xbyak::Reg64 &reg_binary_inj_param_ = abi_param1;
#endif

    const Xbyak::Reg64 &reg_param = abi_param1;
    const Xbyak::Reg64 &reg_stack_frame_ = rbp;
    const Xbyak::Reg64 &reg_dst = rdx;
    const Xbyak::Reg64 &reg_acc = rax;
    const Xbyak::Reg64 &reg_bias = rbx;
    const Xbyak::Reg64 &reg_scales = rsi;

    const Xbyak::Reg64 &reg_oc = r13;
    const Xbyak::Reg64 &reg_len = r8;
    const Xbyak::Reg64 &reg_tmp = rcx; // intentional for shifting purposes
    const Xbyak::Reg64 &reg_tail = reg_tmp;
    const Xbyak::Reg64 &reg_oc_offset = r9;
    const Xbyak::Reg64 &reg_rem_mask = r10;
    const Xbyak::Opmask &kreg_rem_mask = k1;
    const Xbyak::Opmask &opmask_binary = k3;
    const Vmm vmm_rem_mask = Vmm(0);
    // register used for temp computation, needs not to be preserved
    const Xbyak::Reg64 &reg_tmp_comp = r15;

    // *mb_stride used only in matmul_pp_kernel && compute_oc_channel_blk()
    const Xbyak::Reg64 &reg_dst_mb_stride = r12;
    const Xbyak::Reg64 &reg_acc_mb_stride = r14;

    // Will be assigned in constructor
    Vmm vreg_zero, vreg_saturation_ubound, vreg_scale, vreg_sum_scale,
            vreg_sum_zp, vreg_dst_zero_points;

    const Xbyak::Reg64 &eltwise_reserved_gpr_ = r11;
    const Xbyak::Opmask &eltwise_reserved_opmask_ = k2;

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(30);
    Xbyak::Reg64 bf16_emu_reserv_4 = reg_tmp_comp;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(31);

    const int default_OC_loop_unroll_ = is_avx512_ ? 4 : 3;
    int max_OC_loop_unroll_ = 13;
    int idx_compute_vreg_start_ = is_avx512_ ? 0 : 1;
    int idx_compute_vreg_max_ = is_avx512_ ? 31 : 15;
    int compute_vregs_per_iter_ = 1;
    int compute_vreg_bias_shift_ = 0;
    int compute_vreg_prev_dst_shift_ = 0;

    const size_t vlen = cpu_isa_traits<isa>::vlen / sizeof(float);
    static constexpr int reg64_size_ = sizeof(int64_t);
    static constexpr int reg_binary_post_op_oc_off_ = 0;
    static constexpr int reg_binary_post_op_offset_ = 1 * reg64_size_;
    static constexpr int reg_binary_post_op_sp_off_ = 2 * reg64_size_;
    static constexpr int reg_origin_dst_ptr_ = 3 * reg64_size_;
    static constexpr int stack_space_needed_ = 4 * reg64_size_;

    bool any_binary_postop_is_no_bcast_type_ = false;
    bool any_binary_postop_is_per_oc_bcast_type_ = false;
    bool any_binary_postop_is_per_oc_sp_bcast_type_ = false;
    bool any_binary_postop_is_oc_bcast_type_ = false;

    int vreg_dst_idx(const int iter) const {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int iter) { return Vmm(vreg_dst_idx(iter)); }

    Vmm vreg_prev_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_
                + compute_vreg_prev_dst_shift_;
        assert(idx <= idx_compute_vreg_max_);
        return Vmm(idx);
    }

    Vmm vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_
                + compute_vreg_bias_shift_;
        assert(idx <= idx_compute_vreg_max_);
        return Vmm(idx);
    }

    Xbyak::Address dst_ptr(const size_t offt) { return ptr[reg_dst + offt]; }

    Xbyak::Address acc_ptr(const size_t offt) { return ptr[reg_acc + offt]; }

    Xbyak::Address bias_ptr(const size_t offt) { return ptr[reg_bias + offt]; }

    Xbyak::Address stack_ptr(const size_t offt) { return ptr[rsp + offt]; }

    Xbyak::Address scale_ptr(const size_t offt) {
        return ptr[reg_scales + offt];
    }

    Xbyak::Address get_address(const arg_t arg_num, const size_t off) {
        switch (arg_num) {
            case arg_t::dst:
            case arg_t::sum: return dst_ptr(off);
            case arg_t::acc: return acc_ptr(off);
            case arg_t::bias: return bias_ptr(off);
            case arg_t::stack: return stack_ptr(off);
            case arg_t::scale: return scale_ptr(off);
            default: assert(!"unsupported arg_num"); break;
        }
        return Xbyak::Address(0);
    }

    Xbyak::Reg64 get_reg_address(const arg_t arg_num) {
        switch (arg_num) {
            case arg_t::dst:
            case arg_t::sum: return reg_dst;
            case arg_t::acc: return reg_acc;
            case arg_t::bias: return reg_bias;
            case arg_t::stack: return rsp;
            case arg_t::scale: return reg_scales;
            default: assert(!"unsupported arg_num"); break;
        }
        return rsp;
    }

    data_type_t get_data_type(const arg_t arg_num) {
        switch (arg_num) {
            case arg_t::dst: return this->dst_data_type_;
            case arg_t::sum: return this->sum_data_type_;
            case arg_t::acc: return this->acc_data_type_;
            case arg_t::bias: return this->bias_data_type_;
            // default for stack or scale operation
            default: return f32;
        }
        return data_type::undef;
    }
};

template <cpu_isa_t isa>
jit_pp_kernel_t<isa>::jit_pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
        const primitive_attr_t *attr, data_type_t bias_dt, data_type_t acc_dt,
        const memory_desc_t *dst_md, bool skip_sum)
    : pp_kernel_t(
            OC, MB, dst_mb_stride, attr, bias_dt, acc_dt, dst_md, skip_sum) {
    assert(IMPLICATION(this->dst_data_type_ == bf16, mayiuse(avx512_core)));
    assert(isa != avx512_common);

    if (this->do_scale_) vreg_scale = Vmm(idx_compute_vreg_start_++);

    if (this->dst_data_type_ == u8) vreg_zero = Vmm(idx_compute_vreg_start_++);
    if (utils::one_of(this->dst_data_type_, u8, s8, s32))
        vreg_saturation_ubound = Vmm(idx_compute_vreg_start_++);

    if (this->do_sum_) {
        compute_vreg_prev_dst_shift_ = compute_vregs_per_iter_++;
        if (this->sum_scale_ != 1.f)
            vreg_sum_scale = Vmm(idx_compute_vreg_start_++);
        if (this->sum_zp_ != 0) vreg_sum_zp = Vmm(idx_compute_vreg_start_++);
    }

    if (this->do_bias()) compute_vreg_bias_shift_ = compute_vregs_per_iter_++;

    if (!attr->zero_points_.has_default_values(DNNL_ARG_DST)) {
        this->do_dst_zero_points_ = true;
        vreg_dst_zero_points = Vmm(idx_compute_vreg_start_++);
    }

    if (this->dst_data_type_ == bf16 && isa != avx512_core_bf16) {
        idx_compute_vreg_max_ = 27;
        bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_reserv_1,
                bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                bf16_emu_reserv_5));
    }

    int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1)
            / compute_vregs_per_iter_;
    max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);
    if (this->do_eltwise_ || this->do_binary_) {
#define PARAM_OFF(field) offsetof(ker_args_t, field)
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static const size_t helper_vmm_idx = is_avx512_ ? 31 : 15;
        static const size_t prelu_helper_vmm_idx = is_avx512_ ? 30 : 0; // todo: [antonvor] check prelu_helper_vmm_idx if is_avx512_ == false
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const auto dst_md_wrapper = memory_desc_wrapper(*dst_md);

        size_t OC_loop, OC_tail;
        if (OC < max_OC_loop_unroll_ * vlen) {
            // Fully unroll small loops
            OC_loop = 0;
            OC_tail = OC;
        } else {
            OC_loop = vlen * default_OC_loop_unroll_;
            OC_tail = OC % OC_loop;
        }
        size_t tail_size = OC_tail % vlen;
        // enable tail processing for runtime load even if there is no tail
        // for the OC
        tail_size = !!tail_size ? tail_size : 1;
        const binary_injector::rhs_arg_static_params_t rhs_arg_static_params {
                helper_vmm_idx, eltwise_reserved_gpr_, r14, preserve_gpr,
                preserve_vmm, PARAM_OFF(post_ops_binary_rhs_arg_vec),
                dst_md_wrapper, tail_size, opmask_binary, reg_tmp,
                use_exact_tail_scalar_bcast, prelu_helper_vmm_idx};
        static const bcast_set_t enabled_bcast_strategy
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial,
                        broadcasting_strategy_t::per_mb_spatial,
                        broadcasting_strategy_t::no_broadcast};
        const binary_injector::static_params_t binary_static_params {
                reg_binary_inj_param_, enabled_bcast_strategy,
                rhs_arg_static_params};
        static constexpr bool save_state = true;
        const eltwise_injector::static_params_t eltwise_static_params {
                save_state, reg_tmp_comp, eltwise_reserved_opmask_};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<inject_isa_>>(this,
                this->post_ops_, binary_static_params, eltwise_static_params);

        using namespace dnnl::impl::cpu::binary_injector_utils;
        std::tie(any_binary_postop_is_no_bcast_type_,
                any_binary_postop_is_per_oc_bcast_type_,
                any_binary_postop_is_per_oc_sp_bcast_type_,
                any_binary_postop_is_oc_bcast_type_)
                = bcast_strategies_present_tup(this->post_ops_.entry_,
                        dst_md_wrapper, broadcasting_strategy_t::no_broadcast,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial,
                        broadcasting_strategy_t::per_mb_spatial);
    }
#undef PARAM_OFF
}

template <cpu_isa_t isa>
template <typename T>
void jit_pp_kernel_t<isa>::advance_binary_postops_per_oc_off(const T &offset) {

    const auto binary_post_op_oc_off_reg = reg_tmp_comp;
    const auto binary_post_op_current_offset_on_stack
            = ptr[rsp + reg_binary_post_op_oc_off_];

    mov(binary_post_op_oc_off_reg, binary_post_op_current_offset_on_stack);
    add(binary_post_op_oc_off_reg, offset);

    if (this->ndims_ == 2) {
        Xbyak::Label end;
        cmp(binary_post_op_oc_off_reg, this->OC_);
        jl(end, T_NEAR);
        xor_(binary_post_op_oc_off_reg, binary_post_op_oc_off_reg);
        L(end);
    }

    mov(binary_post_op_current_offset_on_stack, binary_post_op_oc_off_reg);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::update_binary_postops_per_tensor_off() {
    // substract dst_origin from current dst and divide it by dst data type
    // size to get the correct offset
    const auto binary_post_op_offset_reg = reg_tmp_comp;
    const auto binary_post_op_current_offset_on_stack
            = ptr[rsp + reg_binary_post_op_offset_];
    mov(binary_post_op_offset_reg, reg_dst);
    sub(binary_post_op_offset_reg, ptr[rsp + reg_origin_dst_ptr_]);
    sar(binary_post_op_offset_reg,
            std::log2(types::data_type_size(get_data_type(arg_t::dst))));
    mov(binary_post_op_current_offset_on_stack, binary_post_op_offset_reg);
}

template <cpu_isa_t isa>
template <typename T>
void jit_pp_kernel_t<isa>::advance_binary_postops_channel_bcast_off(
        const T &offset) {

    const auto binary_post_op_offset_reg = reg_tmp_comp;
    const auto binary_post_op_current_offset_on_stack
            = ptr[rsp + reg_binary_post_op_sp_off_];
    mov(binary_post_op_offset_reg, binary_post_op_current_offset_on_stack);
    add(binary_post_op_offset_reg, offset);
    mov(binary_post_op_current_offset_on_stack, binary_post_op_offset_reg);
}

/*
 * Advance binary postops offsets with per_tensor_offset passed as plain value
 * type (const offset value).
 */
template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::advance_binary_postops_off(const size_t offset) {
    if (offset) {
        if (any_binary_postop_is_per_oc_bcast_type_)
            advance_binary_postops_per_oc_off(offset);
        if (any_binary_postop_is_no_bcast_type_)
            update_binary_postops_per_tensor_off();
        if (any_binary_postop_is_oc_bcast_type_)
            advance_binary_postops_channel_bcast_off(offset);
    }
}

/*
 * Advance binary postops offsets with per_tensor_offset passed in Reg64.
 */
template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::advance_binary_postops_off(
        const Xbyak::Reg64 reg_offset) {
    if (any_binary_postop_is_per_oc_bcast_type_)
        advance_binary_postops_per_oc_off(reg_offset);
    if (any_binary_postop_is_no_bcast_type_)
        update_binary_postops_per_tensor_off();
    if (any_binary_postop_is_oc_bcast_type_)
        advance_binary_postops_channel_bcast_off(reg_offset);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::apply_postops(const bool apply_mask,
        const int vmm_idx, const size_t offset, const bool runtime_tail_mask) {
    if (this->do_eltwise_ || this->do_binary_) {
        if (this->do_binary_) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            if (any_binary_postop_is_per_oc_bcast_type_
                    || any_binary_postop_is_per_oc_sp_bcast_type_) {
                const auto oc_off_oprnd = reg_tmp_comp;
                mov(oc_off_oprnd, ptr[rsp + reg_binary_post_op_oc_off_]);
                rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                        vmm_idx, oc_off_oprnd);
                if (any_binary_postop_is_per_oc_bcast_type_)
                    rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                            vmm_idx, static_cast<int>(offset));
            }

            if (apply_mask) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            rhs_arg_params.tail_load_mode = runtime_tail_mask
                    ? binary_injector::tail_lode_mode_t::DYNAMIC
                    : binary_injector::tail_lode_mode_t::DEFAULT;

            if (any_binary_postop_is_no_bcast_type_) {
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, static_cast<int>(offset));
                rhs_arg_params.vmm_idx_to_out_elem_off_addr.emplace(vmm_idx,
                        ptr[reg_stack_frame_ - stack_space_needed_
                                + reg_binary_post_op_offset_]);
            }
            if (any_binary_postop_is_oc_bcast_type_) {
                rhs_arg_params.vmm_idx_to_sp_elem_off_val.emplace(
                        vmm_idx, static_cast<int>(offset));
                rhs_arg_params.vmm_idx_to_sp_elem_off_addr.emplace(vmm_idx,
                        ptr[reg_stack_frame_ - stack_space_needed_
                                + reg_binary_post_op_sp_off_]);
            }
            postops_injector_->compute_vector(vmm_idx, rhs_arg_params);
        } else
            postops_injector_->compute_vector(vmm_idx);
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::prepare_mask(const size_t tail) {
    assert(tail > 0 && tail <= vlen - 1);
    if (is_avx512_) {
        const size_t tail_mask = (1 << tail) - 1;
        mov(reg_tmp, tail_mask);
        kmovq(kreg_rem_mask, reg_tmp);
    } else if (isa == avx2) {
        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - tail]));
        vmovups(vmm_rem_mask, ptr[reg_tmp]);
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::load_no_tail(
        const Vmm &v, Xbyak::Address op, const data_type_t dt) {
    using namespace data_type;
    switch (dt) {
        case s8: uni_vpmovsxbd(v, op); break;
        case u8: uni_vpmovzxbd(v, op); break;
        case s32:
        case f32: uni_vmovups(v, op); break;
        case bf16:
            vpmovzxwd(v, op);
            vpslld(v, v, 0x10);
            break;
        default: assert(!"unimplemented");
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::load_tail(const Vmm &v, const arg_t arg_num,
        const size_t off, const data_type_t dt, const size_t tail) {
    using namespace data_type;
    if (is_avx512_) {
        auto v_dst = tail ? v | kreg_rem_mask : v;
        load_no_tail(v_dst, get_address(arg_num, off), dt);
    } else {
        if (utils::one_of(dt, s8, u8)) {
            const Xbyak::Xmm x = Xbyak::Xmm(v.getIdx());
            for (size_t i = 0; i < tail; i++)
                uni_vpinsrb(x, x, get_address(arg_num, i + off), i);
            if (dt == s8)
                uni_vpmovsxbd(v, v);
            else
                uni_vpmovzxbd(v, v);
        } else {
            const bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
            if (is_ymm) {
                vmaskmovps(v, vmm_rem_mask, get_address(arg_num, off));
            } else {
                const size_t dt_size = types::data_type_size(f32);
                for (size_t i = 0; i < tail; i++)
                    uni_vpinsrd(
                            v, v, get_address(arg_num, i * dt_size + off), i);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::load_and_cvt(const Vmm &v, const arg_t arg_num,
        const size_t off, const size_t tail, bool do_cvt) {
    using namespace data_type;
    const data_type_t dt = get_data_type(arg_num);
    if (tail)
        load_tail(v, arg_num, off, dt, tail);
    else
        load_no_tail(v, get_address(arg_num, off), dt);

    if (do_cvt && utils::one_of(dt, u8, s8, s32)) uni_vcvtdq2ps(v, v);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::cvt_and_store(const Xbyak::Zmm &v,
        const arg_t arg_num, const size_t off, const size_t tail) {
    using namespace data_type;
    const data_type_t dt = get_data_type(arg_num);
    if (!utils::one_of(dt, f32, bf16)) {
        Vmm vreg = Vmm(v.getIdx()); // in case of use Ymm for bf16
        saturate_f32(vreg, vreg_zero, vreg_saturation_ubound, dt);
        vcvtps2dq(v, v);
    } else if (dt == bf16) {
        if (isa == avx512_core_bf16)
            vcvtneps2bf16(Ymm(v.getIdx()), v);
        else
            bf16_emu_->vcvtneps2bf16(Ymm(v.getIdx()), v);
    }

    auto v_src = tail ? v | kreg_rem_mask : v;
    const Xbyak::Address dst = get_address(arg_num, off);
    switch (dt) {
        case s8: vpmovsdb(dst, v_src); break;
        case u8: vpmovusdb(dst, v_src); break;
        case f32:
        case s32: uni_vmovups(dst, v_src); break;
        case bf16:
            vmovdqu16(dst,
                    tail ? Ymm(v.getIdx()) | kreg_rem_mask : Ymm(v.getIdx()));
            break;
        default: assert(!"unimplemented");
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::cvt_and_store(const Xbyak::Ymm &v,
        const arg_t arg_num, const size_t off, const size_t tail) {
    using namespace data_type;
    const data_type_t dt = get_data_type(arg_num);
    const Xbyak::Address dst = get_address(arg_num, off);
    const Xbyak::Xmm x = Xbyak::Xmm(v.getIdx());
    if (dt == bf16) {
        // use Zmm implementation for bf16 with Ymm
        cvt_and_store(Xbyak::Zmm(v.getIdx()), arg_num, off, tail);
        return;
    } else if (utils::one_of(dt, s8, u8, s32)) {
        saturate_f32(v, vreg_zero, vreg_saturation_ubound, dt);
        vcvtps2dq(v, v);

        if (dt != s32) {
            // v = { 8x32 }
            vpackssdw(v, v, vreg_zero);
            // v = { 4x16, 0, 4x16, 0 }
            vpermq(v, v, 0x58);
            // v =  { 8x16, 0 }
            if (dt == s8)
                vpacksswb(v, v, vreg_zero);
            else
                vpackuswb(v, v, vreg_zero);
        }
    }

    if (tail) {
        switch (dt) {
            case s8:
            case u8:
                for (size_t i = 0; i < tail; i++)
                    vpextrb(get_address(arg_num, off + i), x, i);
                break;
            case f32:
            case s32: vmaskmovps(dst, vmm_rem_mask, v); break;
            default: assert(!"unimplemented");
        }
    } else {
        switch (dt) {
            case s8:
            case u8: vmovq(dst, x); break;
            case f32:
            case s32: vmovups(dst, v); break;
            default: assert(!"unimplemented");
        }
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::cvt_and_store(const Xbyak::Xmm &v,
        const arg_t arg_num, const size_t off, const size_t tail) {
    using namespace data_type;
    const data_type_t dt = get_data_type(arg_num);
    const Xbyak::Address dst = get_address(arg_num, off);
    if (utils::one_of(dt, s8, u8, s32)) {
        saturate_f32(v, vreg_zero, vreg_saturation_ubound, dt);
        uni_vcvtps2dq(v, v);

        if (dt != s32) {
            // v = { 8x32 }
            uni_vpackssdw(v, v, vreg_zero);
            // v = { 4x16, 0}
            if (dt == s8)
                uni_vpacksswb(v, v, vreg_zero);
            else
                uni_vpackuswb(v, v, vreg_zero);
        }
    }

    if (tail) {
        switch (dt) {
            case s8:
            case u8:
                for (size_t i = 0; i < tail; i++)
                    uni_vpextrb(get_address(arg_num, off + i), v, i);
                break;
            case f32:
            case s32: {
                const size_t dt_size = types::data_type_size(f32);
                for (size_t i = 0; i < tail; i++)
                    uni_vpextrd(get_address(arg_num, off + i * dt_size), v, i);
            } break;
            default: assert(!"unimplemented");
        }
    } else {
        switch (dt) {
            case s8:
            case u8: uni_vmovd(dst, v); break;
            case f32:
            case s32: uni_vmovups(dst, v); break;
            default: assert(!"unimplemented");
        }
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::runtime_tail_load_cvt(
        const Vmm &v, const arg_t arg_num, const size_t off, bool cvt) {
    assert(!is_avx512_);
    const data_type_t dt = get_data_type(arg_num);
    const bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    const Xbyak::Xmm x = Xbyak::Xmm(v.getIdx());
    const Xbyak::Ymm y = Xbyak::Ymm(v.getIdx());
    const Xbyak::Reg64 &reg_addr = get_reg_address(arg_num);

    auto runtime_tail_load = [&](int load_size) {
        if (is_ymm)
            load_data(dt, y, reg_addr, off, load_size);
        else
            load_data(dt, x, reg_addr, off, load_size);
    };

    runtime_tail_process<Vmm>(reg_tail, reg_rem_mask, runtime_tail_load);

    if (cvt && utils::one_of(dt, u8, s8, s32)) uni_vcvtdq2ps(v, v);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::runtime_tail_cvt_store(
        const Vmm &v, const arg_t arg_num, const size_t off) {
    assert(!is_avx512_);
    const data_type_t dt = get_data_type(arg_num);
    const bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
    const Xbyak::Xmm x = Xbyak::Xmm(v.getIdx());
    const Xbyak::Ymm y = Xbyak::Ymm(v.getIdx());
    const Xbyak::Reg64 &reg_addr = get_reg_address(arg_num);

    if (utils::one_of(dt, u8, s8, s32)) {
        saturate_f32(v, vreg_zero, vreg_saturation_ubound, dt);
        uni_vcvtps2dq(v, v);
    }

    auto runtime_tail_store = [&](int store_size) {
        if (is_ymm)
            store_data(dt, y, reg_addr, off, store_size);
        else
            store_data(dt, x, reg_addr, off, store_size);
    };

    runtime_tail_process<Vmm>(reg_tail, reg_rem_mask, runtime_tail_store);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::data_copy(const Vmm &v, const arg_t arg_num,
        const size_t off, data_op_t data_op, const size_t tail,
        const bool is_needed_runtime_tail_process, const bool do_cvt) {
    if (data_op == data_op_t::load) {
        if (is_needed_runtime_tail_process)
            runtime_tail_load_cvt(v, arg_num, off, do_cvt);
        else
            load_and_cvt(v, arg_num, off, tail, do_cvt);
    } else {
        if (is_needed_runtime_tail_process)
            runtime_tail_cvt_store(v, arg_num, off);
        else
            cvt_and_store(v, arg_num, off, tail);
    }
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::compute_oc_channel_blk() {
    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store

    auto compute = [&](size_t offset, int idx, bool runtime_tail_mask,
                           int tail = 0) {
        const bool is_needed_runtime_tail_process
                = runtime_tail_mask && tail && !is_avx512_;

        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            data_copy(vreg_scale, arg_t::scale, offset * sizeof(float),
                    data_op_t::load, tail, is_needed_runtime_tail_process,
                    false);

        if (this->do_binary_ && tail && is_avx512_)
            kmovq(opmask_binary, kreg_rem_mask);

        const int dst_idx = vreg_dst_idx(idx);
        auto vreg_dst_ = Vmm(dst_idx);
        data_copy(vreg_dst_, arg_t::acc, offset * this->acc_data_type_size_,
                data_op_t::load, tail, is_needed_runtime_tail_process);

        if (this->do_bias()) {
            auto vreg_bias_ = vreg_bias(idx);
            data_copy(vreg_bias_, arg_t::bias,
                    offset * this->bias_data_type_size_, data_op_t::load, tail,
                    is_needed_runtime_tail_process);
            uni_vaddps(vreg_dst_, vreg_dst_, vreg_bias_);
        }

        if (this->do_scale_) uni_vmulps(vreg_dst_, vreg_dst_, vreg_scale);

        if (this->do_sum_) {
            auto vreg_prev_dst_ = vreg_prev_dst(idx);
            data_copy(vreg_prev_dst_, arg_t::sum,
                    offset * this->dst_data_type_size_, data_op_t::load, tail,
                    is_needed_runtime_tail_process);
            if (this->sum_zp_ != 0)
                uni_vsubps(vreg_prev_dst_, vreg_prev_dst_, vreg_sum_zp);
            if (this->sum_scale_ != 1.f)
                uni_vfmadd231ps(vreg_dst_, vreg_prev_dst_, vreg_sum_scale);
            else
                uni_vaddps(vreg_dst_, vreg_dst_, vreg_prev_dst_);
        }

        apply_postops(!!tail, dst_idx, offset, is_needed_runtime_tail_process);

        if (this->do_dst_zero_points_)
            uni_vaddps(vreg_dst_, vreg_dst_, vreg_dst_zero_points);

        data_copy(vreg_dst_, arg_t::dst, offset * this->dst_data_type_size_,
                data_op_t::store, tail, is_needed_runtime_tail_process);
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * this->dst_data_type_size_);
        add(reg_acc, offset * this->acc_data_type_size_);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            add(reg_scales, offset * sizeof(float));
        if (this->do_bias()) add(reg_bias, offset * this->bias_data_type_size_);
        if (this->do_binary_) { advance_binary_postops_off(offset); }
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](const Reg64 &offset) {
        lea(reg_dst, ptr[reg_dst + offset * this->dst_data_type_size_]);
        lea(reg_acc, ptr[reg_acc + offset * this->acc_data_type_size_]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        if (this->do_bias())
            lea(reg_bias, ptr[reg_bias + offset * this->bias_data_type_size_]);
        if (this->do_binary_) advance_binary_postops_off(offset);
    };

    // incase of non-trivial dst_mb_strides, fixup the reg_dst and reg_acc
    auto maybe_advance_mb_stride = [&]() {
        if (!this->has_trivial_mb_stride()) {
            lea(reg_dst,
                    ptr[reg_dst
                            + reg_dst_mb_stride * this->dst_data_type_size_]);
            lea(reg_acc,
                    ptr[reg_acc
                            + reg_acc_mb_stride * this->acc_data_type_size_]);
        }
        if (this->do_binary_ && any_binary_postop_is_no_bcast_type_)
            update_binary_postops_per_tensor_off();
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        neg(reg_oc);
        if (this->do_bias())
            lea(reg_bias, ptr[reg_bias + reg_oc * this->bias_data_type_size_]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + reg_oc * sizeof(float)]);

        neg(reg_oc);
    };

    // Process one row of reg_tmp elements
    auto process_runtime_oc = [&]() {
        Label l_loop, l_loop_tail, l_loop_end;
        cmp(reg_tmp, vlen);
        jl(l_loop_tail, T_NEAR);

        L(l_loop);
        {
            compute(0, 0, true);
            advance_ptrs_imm(vlen);

            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(l_loop, T_NEAR);
        }

        L(l_loop_tail);
        cmp(reg_tmp, 0);
        je(l_loop_end, T_NEAR);

        if (is_avx512_) {
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // cl == reg_tmp because reg_tmp <= vlen here
            sub(reg_rem_mask, 1);
            kmovq(kreg_rem_mask, reg_rem_mask);
        }
        // tail size does not matter for runtime load
        compute(0, 0, true, true);
        advance_ptrs_reg(reg_tmp);

        L(l_loop_end);
    };

    //      <-------------------- OC ------------------------------->
    //
    // ^    +....................+----------------------------------+
    // |    :   not accessed     |          Prologue loop           |
    // |    +--------------------+----------------------------------+
    //      |                                                       |
    // M    |                 Main loop (unrolled)                  |
    // B    |                                                       |
    //      +--------------------------------+----------------------+
    // |    |       Epilogue loop            |      not accessed    :
    // v    +--------------------------------+......................+

    if (this->dst_data_type_ == bf16 && isa != avx512_core_bf16)
        bf16_emu_->init_vcvtneps2bf16();

    // Prologue loop
    Label l_prologue_end;
    cmp(reg_oc_offset, 0);
    je(l_prologue_end, T_NEAR);
    {
        mov(reg_tmp, reg_oc);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);
        process_runtime_oc();
        rewind_ptrs();
        maybe_advance_mb_stride();
    }
    L(l_prologue_end);

    // Main loop
    Label l_main_loop_end;
    cmp(reg_len, reg_oc);
    jle(l_main_loop_end, T_NEAR);
    if (this->runtime_oc()) {
        Label l_main_loop;
        L(l_main_loop);
        {
            mov(reg_tmp, reg_oc);

            process_runtime_oc();
            rewind_ptrs();

            sub(reg_len, reg_oc);
            maybe_advance_mb_stride();
            cmp(reg_len, reg_oc);
            jge(l_main_loop, T_NEAR);
        }
    } else {
        Label l_main_loop;
        L(l_main_loop);
        {
            size_t OC_loop, OC_tail;
            if (this->OC_ < max_OC_loop_unroll_ * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = this->OC_;
            } else {
                OC_loop = vlen * default_OC_loop_unroll_;
                OC_tail = this->OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            const int vlen_tail = OC_tail % vlen;
            if (vlen_tail) prepare_mask(vlen_tail);

            if (OC_loop) {
                mov(reg_tmp, utils::rnd_dn(this->OC_, OC_loop));
                Label l_oc_loop;
                L(l_oc_loop);
                {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(l_oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    const bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, false,
                            use_mask ? vlen_tail : 0);
                }
                advance_ptrs_imm(OC_tail);
            }

            if (any_binary_postop_is_per_oc_sp_bcast_type_
                    && this->ndims_ <= 3) {
                static constexpr size_t offset_oc_spatial = 1;
                advance_binary_postops_per_oc_off(offset_oc_spatial);
            }

            rewind_ptrs();
            sub(reg_len, reg_oc);
            maybe_advance_mb_stride();
            cmp(reg_len, reg_oc);
            jge(l_main_loop, T_NEAR);
        }
    }
    L(l_main_loop_end);

    // Epilogue loop
    Label l_epilogue_end;
    cmp(reg_len, 0);
    je(l_epilogue_end, T_NEAR);
    {
        mov(reg_tmp, reg_len);
        process_runtime_oc();
    }
    L(l_epilogue_end);
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::compute_mb_blk() {
    auto compute = [&](int tail, bool runtime_tail = false) {
        auto vmm_bias = vreg_bias(0);
        auto vmm_dst = vreg_dst(0);
        assert(utils::one_of(this->acc_data_type_, s32, f32));
        data_copy(vmm_dst, arg_t::acc, 0, data_op_t::load, tail, runtime_tail);
        uni_vaddps(vmm_dst, vmm_dst, vmm_bias);
        data_copy(vmm_dst, arg_t::dst, 0, data_op_t::store, tail, runtime_tail);
    };

    Label mb_main_loop, end_main_loop;

    bool expl_broadcast
            = this->OC_ == 1 && utils::one_of(this->bias_data_type_, s32, f32);
    size_t mb_step = vlen / this->OC_;
    size_t mb_tail = this->MB_ % mb_step;
    size_t mb_oc_blk = mb_step * this->OC_;
    size_t tail_size = mb_oc_blk % vlen;
    auto vmm_bias = vreg_bias(0);

    if (this->dst_data_type_ == bf16 && isa != avx512_core_bf16)
        bf16_emu_->init_vcvtneps2bf16();

    if (expl_broadcast) {
        // when OC == 1 bias can be loaded directly into simd
        switch (this->bias_data_type_) {
            case s32: uni_vpbroadcastd(vmm_bias, ptr[reg_bias]); break;
            case f32: uni_vbroadcastss(vmm_bias, ptr[reg_bias]); break;
            // TODO: enable broadcast for other data types
            default: assert(!"unimplemented");
        }
    } else {
        // prepare bias data for simd computation
        prepare_mask(this->OC_); // this->OC will never be larger than vlen / 2
        load_and_cvt(vmm_bias, arg_t::bias, 0, this->OC_, false);

        // write repeated MB*OC entries into stack
        sub(rsp, mb_oc_blk * sizeof(uint32_t));
        for (size_t i = 0; i < mb_step; ++i)
            cvt_and_store(vmm_bias, arg_t::stack,
                    i * this->OC_ * sizeof(uint32_t), this->OC_);

        // load into simd
        if (tail_size) prepare_mask(tail_size);
        load_and_cvt(vmm_bias, arg_t::stack, 0, tail_size, false);
    }
    if (utils::one_of(this->bias_data_type_, u8, s8, s32))
        uni_vcvtdq2ps(vmm_bias, vmm_bias);
    L(mb_main_loop);
    {
        cmp(reg_len, mb_oc_blk);
        jl(end_main_loop, T_NEAR);

        compute(!expl_broadcast ? tail_size : 0);
        add(reg_dst, mb_oc_blk * this->dst_data_type_size_);
        add(reg_acc, mb_oc_blk * this->acc_data_type_size_);
        sub(reg_len, mb_oc_blk);
        jmp(mb_main_loop, T_NEAR);
    }
    L(end_main_loop);

    if (mb_tail > 0) {
        Label mb_tail_loop, runtime_tail, end_runtime_tail;
        tail_size = (mb_tail * this->OC_);
        if (tail_size) prepare_mask(tail_size);
        L(mb_tail_loop);
        {
            cmp(reg_len, tail_size);
            jl(runtime_tail, T_NEAR);
            compute(tail_size);
            add(reg_dst, tail_size * this->dst_data_type_size_);
            add(reg_acc, tail_size * this->acc_data_type_size_);
            sub(reg_len, tail_size);
            jmp(mb_tail_loop, T_NEAR);
        }
        // Load tail in runtime if len < mb_tail * oc
        L(runtime_tail);
        {
            cmp(reg_len, 0);
            jle(end_runtime_tail, T_NEAR);
            mov(reg_tail, reg_len); // save tail
            if (is_avx512_) {
                mov(reg_rem_mask, 1);
                shl(reg_rem_mask, cl); // cl == last 8 bits of reg_tail
                sub(reg_rem_mask, 1);
                kmovq(kreg_rem_mask, reg_rem_mask);
            }
            compute(tail_size, !is_avx512_);
        }
        L(end_runtime_tail);
    }

    if (!expl_broadcast) add(rsp, mb_oc_blk * sizeof(uint32_t));
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::generate() {
    preamble();

#ifdef _WIN32
    // binary postops injector needs params held (in case of WIN32)
    // in rcx register that is also used as a temp reg, so the pointer to
    // params needs to be stored in extra reg
    if (this->do_binary_) mov(reg_binary_inj_param_, param1);
#endif

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    if (this->do_scale_) mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    if (this->do_dst_zero_points_) {
        // use reg_oc as a temporary one (alas, reg_tmp = reg_param on Windows)
        mov(reg_oc, ptr[reg_param + PARAM_OFF(dst_zero_points)]);
        uni_vbroadcastss(vreg_dst_zero_points, ptr[reg_oc]);
    }
    if (this->runtime_oc())
        mov(reg_oc, ptr[reg_param + PARAM_OFF(oc)]);
    else
        mov(reg_oc, this->OC_);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    if (this->do_binary_) {
        mov(reg_stack_frame_, rsp);
        sub(rsp, stack_space_needed_);
        if (any_binary_postop_is_per_oc_sp_bcast_type_
                || any_binary_postop_is_per_oc_bcast_type_) {
            mov(reg_tmp_comp, ptr[reg_param + PARAM_OFF(dim1_off)]);
            mov(ptr[rsp + reg_binary_post_op_oc_off_], reg_tmp_comp);
        }
        if (any_binary_postop_is_no_bcast_type_) {
            // store origin dst pointer to calculate proper binary src1 offset
            mov(reg_tmp_comp, ptr[reg_param + PARAM_OFF(dst_orig)]);
            mov(ptr[rsp + reg_origin_dst_ptr_], reg_tmp_comp);
            // init offset
            update_binary_postops_per_tensor_off();
        }
        if (any_binary_postop_is_oc_bcast_type_) {
            // initialize binary post_ops no bcast offset accumulator
            mov(reg_tmp_comp,
                    ptr[reg_param + PARAM_OFF(first_mb_matrix_addr_off)]);
            mov(ptr[rsp + reg_binary_post_op_sp_off_], reg_tmp_comp);
        }
    }
    if (this->do_scale_ && this->scale_idx_mult_ == 0)
        uni_vbroadcastss(vreg_scale, dword[reg_scales]);
    if (!this->has_trivial_mb_stride()) {
        mov(reg_dst_mb_stride, ptr[reg_param + PARAM_OFF(dst_mb_stride)]);
        sub(reg_dst_mb_stride, reg_oc);
        // if dst and acc point to same address (in-place), then strides must be
        // similar, else assume acc buffer is dense.
        xor_(reg_acc_mb_stride, reg_acc_mb_stride);
        cmp(reg_dst, reg_acc);
        cmove(reg_acc_mb_stride, reg_dst_mb_stride);
    }
#undef PARAM_OFF

    if (this->do_sum_) {
        if (this->sum_scale_ != 1.f) {
            mov(reg_tmp, float2int(this->sum_scale_));
            auto xreg_sum_scale = Xmm(vreg_sum_scale.getIdx());
            uni_vmovq(xreg_sum_scale, reg_tmp);
            uni_vbroadcastss(vreg_sum_scale, xreg_sum_scale);
        }
        if (this->sum_zp_ != 0) {
            mov(reg_tmp, this->sum_zp_);
            auto xreg_sum_zp = Xmm(vreg_sum_zp.getIdx());
            uni_vmovq(xreg_sum_zp, reg_tmp);
            uni_vbroadcastss(vreg_sum_zp, xreg_sum_zp);
            uni_vcvtdq2ps(vreg_sum_zp, vreg_sum_zp);
        }
    }

    init_saturate_f32(vreg_zero, vreg_saturation_ubound, reg_tmp_comp, f32,
            this->dst_data_type_);

    // at least 2 blocks of mb within vlen
    bool dim_restrict = !this->runtime_oc() && !this->runtime_mb()
            && (this->OC_ <= vlen / 2) && (this->MB_ >= vlen);
    bool supported_postops = this->do_scale_ || (this->post_ops_.len() > 0) || this->do_dst_zero_points_;
    if (this->do_bias() && !supported_postops && dim_restrict
            && this->has_trivial_mb_stride()) {
        this->mb_blk_kernel_ = true;
        compute_mb_blk();
    } else {
        compute_oc_channel_blk();
    }

    if (this->do_binary_) add(rsp, stack_space_needed_);
    postamble();

    if (this->do_eltwise_) postops_injector_->prepare_table();
}

template <cpu_isa_t isa>
void jit_pp_kernel_t<isa>::operator()(void *dst, const void *acc,
        const char *bias, const float *scales, size_t start,
        size_t dst_logical_off, size_t dim1_off, size_t end, size_t runtime_oc,
        dim_t dst_mb_stride, const float *dst_zero_points,
        const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
        size_t first_mb_matrix_addr_off, const exec_ctx_t & /* ctx */,
        const memory_desc_t & /* dst_md */) const {

    if (end <= start) return;
    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;

    ker_args_t args;
    size_t oc_offset = start % OC;
    if (this->has_trivial_mb_stride()) {
        args.dst = static_cast<char *>(dst) + this->dst_data_type_size_ * start;
        args.acc = static_cast<const char *>(acc)
                + this->acc_data_type_size_ * start;
    } else {
        const dim_t offt = (start / OC) * dst_mb_stride + oc_offset;
        args.dst = static_cast<char *>(dst) + this->dst_data_type_size_ * offt;
        // if dst and acc point to same address (inplace), then strides
        // must be similar, else assume acc buffer is dense.
        const auto stride = dst == acc ? offt : start;
        args.acc = static_cast<const char *>(acc)
                + this->acc_data_type_size_ * stride;
    }
    args.bias = bias + oc_offset * this->bias_data_type_size_;
    args.scales = scales + this->scale_idx_mult_ * oc_offset;
    args.dst_zero_points = dst_zero_points;
    args.oc = OC;
    args.len = end - start;
    args.oc_offset = oc_offset;
    args.dst_logical_off = dst_logical_off;
    args.dim1_off = dim1_off;
    args.dst_mb_stride = dst_mb_stride;
    args.first_mb_matrix_addr_off = first_mb_matrix_addr_off;

    args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
    args.dst_orig = dst_orig;
    jit_generator::operator()(&args);
}

pp_kernel_t *jit_pp_kernel_create(size_t OC, size_t MB, dim_t dst_mb_stride,
        const primitive_attr_t *attr, data_type_t bias_dt, data_type_t acc_dt,
        const memory_desc_t *dst_md, bool skip_sum) {
    if (mayiuse(avx512_core_bf16)) {
        return new jit_pp_kernel_t<avx512_core_bf16>(
                OC, MB, dst_mb_stride, attr, bias_dt, acc_dt, dst_md, skip_sum);
    } else if (mayiuse(avx512_core)) {
        return new jit_pp_kernel_t<avx512_core>(
                OC, MB, dst_mb_stride, attr, bias_dt, acc_dt, dst_md, skip_sum);
    } else if (mayiuse(avx2)) {
        return new jit_pp_kernel_t<avx2>(
                OC, MB, dst_mb_stride, attr, bias_dt, acc_dt, dst_md, skip_sum);
    } else if (mayiuse(sse41)) {
        return new jit_pp_kernel_t<sse41>(
                OC, MB, dst_mb_stride, attr, bias_dt, acc_dt, dst_md, skip_sum);
    } else {
        return nullptr;
    }
}

} // namespace inner_product_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
