/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <type_traits>

#include "cpu/x64/prelu/jit_uni_prelu_backward_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_prelu_backward_kernel_t::jit_prelu_backward_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa, const int vlen,
        const size_t number_vmm_single_compute)
    : jit_prelu_base_kernel_t(isa, vlen,
            prelu::get_bcast_type(memory_desc_wrapper(pd->diff_src_md(0)),
                    memory_desc_wrapper(pd->diff_weights_md(0))),
            memory_desc_wrapper(pd->diff_src_md(0)), number_vmm_single_compute)
    , pd_(pd)
    , src_dt_(pd->src_md(0)->data_type)
    , wei_dt_(pd->weights_md(0)->data_type)
    , diff_src_dt_(pd->diff_src_md(0)->data_type)
    , diff_dst_dt_(pd->diff_dst_md(0)->data_type)
    , diff_wei_dt_(bcast_ == prelu::bcast::full
                      ? pd->diff_weights_md(0)->data_type
                      : data_type::f32)
    , diff_src_block_tail_(prelu::get_block_tail_size(pd->diff_src_md(0)))
    , diff_wei_block_tail_(prelu::get_block_tail_size(pd->diff_weights_md(0))) {
}

#define PARAM_OFF(x) offsetof(call_params_t, x)

void jit_prelu_backward_kernel_t::load_kernel_call_params() {
    mov(reg_src_, ptr[abi_param1 + PARAM_OFF(src)]);
    mov(reg_weights_, ptr[abi_param1 + PARAM_OFF(weights)]);
    mov(reg_src_diff_, ptr[abi_param1 + PARAM_OFF(src_diff)]);
    mov(reg_weights_diff_, ptr[abi_param1 + PARAM_OFF(weights_diff)]);
    mov(reg_dst_diff_, ptr[abi_param1 + PARAM_OFF(dst_diff)]);
    mov(reg_data_size_, ptr[abi_param1 + PARAM_OFF(compute_data_size)]);
}

#undef PARAM_OFF

Xbyak::Address jit_prelu_backward_kernel_t::data_ptr(int arg_num, size_t offt) {
    const auto get_addr
            = [&](const Xbyak::Reg64 &reg_base, const data_type_t dt) {
                  const auto dt_size = types::data_type_size(dt);
                  return ptr[reg_base + reg_offset_ * dt_size + offt * dt_size];
              };

    switch (arg_num) {
        case DNNL_ARG_SRC: return get_addr(reg_src_, src_dt_);
        case DNNL_ARG_WEIGHTS: return get_addr(reg_weights_, wei_dt_);
        case DNNL_ARG_DIFF_SRC: return get_addr(reg_src_diff_, diff_src_dt_);
        case DNNL_ARG_DIFF_WEIGHTS:
            return get_addr(reg_weights_diff_, diff_wei_dt_);
        case DNNL_ARG_DIFF_DST: return get_addr(reg_dst_diff_, diff_dst_dt_);

        default: assert(!"unsupported arg_num"); break;
    }
    return Xbyak::Address(0);
}

bool jit_prelu_backward_kernel_t::any_tensor_bf16() const {
    return utils::one_of(data_type::bf16, src_dt_, wei_dt_, diff_src_dt_,
            diff_dst_dt_, diff_wei_dt_);
}

template <typename Vmm>
jit_uni_prelu_backward_kernel_t<Vmm>::jit_uni_prelu_backward_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa)
    : jit_prelu_backward_kernel_t(pd, isa, prelu::vmm_traits_t<Vmm>::vlen,
            std::is_same<Vmm, Xbyak::Zmm>::value ? 4u : 6u)
    , saturation_needed_diff_src_(utils::one_of(
              diff_src_dt_, data_type::u8, data_type::s8, data_type::s32))
    , saturation_needed_diff_weights_(utils::one_of(
              diff_wei_dt_, data_type::u8, data_type::s8, data_type::s32))
    , vmm_zeros_(reserve_vmm())
    , saturation_ubound_diff_src_(
              saturation_needed_diff_src_ ? reserve_vmm() : 0)
    , saturation_ubound_diff_weights_(saturation_needed_diff_weights_
                      ? (diff_wei_dt_ == diff_src_dt_
                                      ? saturation_ubound_diff_src_.getIdx()
                                      : reserve_vmm())
                      : 0)
    , tail_vmm_mask_(
              tail_size_ && utils::one_of(isa, avx, avx2) ? reserve_vmm() : 0)
    , vmm_ones_(reserve_vmm())
    , weights_const_vmm_(utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                                 prelu::bcast::per_oc_blocked)
                      ? reserve_vmm()
                      : 0)
    , weights_diff_acc_vmm_(
              utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                      prelu::bcast::per_oc_blocked)
                      ? reserve_vmm()
                      : 0)
    , io_(this, isa,
              {src_dt_, wei_dt_, diff_src_dt_, diff_dst_dt_, diff_wei_dt_}, {},
              io::io_tail_conf_t {simd_w_, tail_size_, tail_opmask_,
                      tail_vmm_mask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {}, create_saturation_vmm_map()) {}

template <typename Vmm>
jit_uni_prelu_backward_kernel_t<Vmm>::~jit_uni_prelu_backward_kernel_t()
        = default;

template <typename Vmm>
void jit_uni_prelu_backward_kernel_t<Vmm>::prepare_kernel_const_vars() {
    uni_vxorps(vmm_zeros_, vmm_zeros_, vmm_zeros_);

    io_.init_bf16();
    if (tail_size_) io_.prepare_tail_mask();
    if (saturation_needed_diff_src_ || saturation_needed_diff_weights_) {
        io_.init_saturate_f32({diff_src_dt_, diff_wei_dt_});
    }
    // load ones
    this->mov(this->reg_tmp_, float2int(1));
    const Xbyak::Xmm xmm_ones_ {vmm_ones_.getIdx()};
    this->uni_vmovq(xmm_ones_, this->reg_tmp_);
    this->uni_vbroadcastss(vmm_ones_, xmm_ones_);

    if (bcast_ == prelu::bcast::per_oc_blocked) {
        io_.at(wei_dt_)->load(
                ptr[reg_weights_], weights_const_vmm_, false /*tail*/);
        vmovups(weights_diff_acc_vmm_, ptr[reg_weights_diff_]);
    } else if (bcast_ == prelu::bcast::per_oc_n_c_spatial) {
        io_.at(wei_dt_)->broadcast(ptr[reg_weights_], weights_const_vmm_);
        uni_vxorps(weights_diff_acc_vmm_, weights_diff_acc_vmm_,
                weights_diff_acc_vmm_);
        uni_vmovss(weights_diff_acc_vmm_, ptr[reg_weights_diff_]);
    }
}

template <typename Vmm>
void jit_uni_prelu_backward_kernel_t<Vmm>::compute_dst(
        size_t unrolling_factor, bool tail) {

    static constexpr size_t dst_diff_idx = 0;
    static constexpr size_t src_idx = 1;
    static constexpr size_t src_le_zero_idx = 2;
    static constexpr size_t src_gt_zero_idx = 3;
    static constexpr size_t weights_diff_idx = 4;
    static constexpr size_t weights_idx = 5;

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {

        const Vmm dst_diff_vmm {get_compute_vmm(dst_diff_idx, unroll_group)};
        const Vmm src_vmm {get_compute_vmm(src_idx, unroll_group)};
        const Vmm src_le_zero_vmm {
                get_compute_vmm(src_le_zero_idx, unroll_group)};
        const Vmm src_gt_zero_vmm {
                get_compute_vmm(src_gt_zero_idx, unroll_group)};
        const Vmm weights_diff_vmm {
                get_compute_vmm(weights_diff_idx, unroll_group)};
        const Vmm weights_vmm {get_compute_vmm(weights_idx, unroll_group)};

        const auto offset = unroll_group * simd_w_;
        io_.at(diff_dst_dt_)
                ->load(data_ptr(DNNL_ARG_DIFF_DST, offset), dst_diff_vmm, tail);
        io_.at(src_dt_)->load(data_ptr(DNNL_ARG_SRC, offset), src_vmm, tail);
        static constexpr int VCMPLEPS = 2;
        uni_vcmpps(src_le_zero_vmm, src_vmm, vmm_zeros_, VCMPLEPS);
        uni_vandps(src_le_zero_vmm, src_le_zero_vmm, vmm_ones_);
        static constexpr int VCMPGTPS = 14;
        uni_vcmpps(src_gt_zero_vmm, src_vmm, vmm_zeros_, VCMPGTPS);
        uni_vandps(src_gt_zero_vmm, src_gt_zero_vmm, vmm_ones_);

        //weights_diff_calculations
        uni_vmulps(weights_diff_vmm, dst_diff_vmm, src_vmm);
        uni_vmulps(weights_diff_vmm, weights_diff_vmm, src_le_zero_vmm);

        //src_diff calculations
        const auto weights_operand = get_or_load_weights(
                data_ptr(DNNL_ARG_WEIGHTS, offset), weights_vmm, tail);
        uni_vfmadd231ps(src_gt_zero_vmm, src_le_zero_vmm, weights_operand);
        const auto &src_diff_vmm = src_gt_zero_vmm;
        uni_vmulps(src_diff_vmm, src_diff_vmm, dst_diff_vmm);
        io_.at(diff_src_dt_)
                ->store(src_diff_vmm, data_ptr(DNNL_ARG_DIFF_SRC, offset),
                        tail);
        if (diff_src_block_tail_ && tail)
            prelu::apply_zero_padding(this, tail_size_, diff_src_dt_,
                    diff_src_block_tail_, reg_src_diff_, nullptr);

        accumulate_weights_diff(weights_diff_vmm, src_gt_zero_vmm,
                data_ptr(DNNL_ARG_DIFF_WEIGHTS, offset), tail);
    }
}

template <>
void jit_uni_prelu_backward_kernel_t<Xbyak::Zmm>::compute_dst(
        size_t unrolling_factor, bool tail) {

    size_t opmask_counter = 2;
    auto get_next_opmask = [opmask_counter]() mutable {
        static constexpr size_t opmask_range_begin = 2;
        static constexpr size_t opmask_range_end = 8;
        const auto opmask = Xbyak::Opmask(opmask_counter++);
        if (opmask_counter == opmask_range_end)
            opmask_counter = opmask_range_begin;
        return opmask;
    };

    static constexpr size_t dst_diff_idx = 0;
    static constexpr size_t src_idx = 1;
    static constexpr size_t weights_diff_idx = 2;
    static constexpr size_t weights_idx = 3;

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {

        const auto offset = unroll_group * simd_w_;
        const Xbyak::Zmm dst_diff_vmm {
                get_compute_vmm(dst_diff_idx, unroll_group)};
        const Xbyak::Zmm src_vmm {get_compute_vmm(src_idx, unroll_group)};

        io_.at(diff_dst_dt_)
                ->load(data_ptr(DNNL_ARG_DIFF_DST, offset), dst_diff_vmm, tail);
        io_.at(src_dt_)->load(data_ptr(DNNL_ARG_SRC, offset), src_vmm, tail);

        const Xbyak::Opmask src_le_zero_opmask = get_next_opmask();
        static constexpr int VCMPLEPS = 2;
        vcmpps(src_le_zero_opmask, src_vmm, vmm_zeros_, VCMPLEPS);
        const Xbyak::Opmask src_gt_zero_vmm_opmask = get_next_opmask();
        static constexpr int VCMPGTPS = 14;
        vcmpps(src_gt_zero_vmm_opmask, src_vmm, vmm_zeros_, VCMPGTPS);

        // //weights_diff_calculations
        const Xbyak::Zmm weights_diff_vmm {
                get_compute_vmm(weights_diff_idx, unroll_group)};
        vmulps(weights_diff_vmm | src_le_zero_opmask | T_z, dst_diff_vmm,
                src_vmm);
        accumulate_weights_diff(weights_diff_vmm, weights_diff_acc_vmm_,
                data_ptr(DNNL_ARG_DIFF_WEIGHTS, offset), tail);

        //src_diff calculations
        const Xbyak::Zmm weights_vmm {
                get_compute_vmm(weights_idx, unroll_group)};
        const auto &src_diff_vmm = weights_vmm;
        const auto weights_operand = get_or_load_weights(
                data_ptr(DNNL_ARG_WEIGHTS, offset), weights_vmm, tail);

        vmovaps(src_diff_vmm | src_le_zero_opmask | T_z, weights_operand);
        vaddps(src_diff_vmm | src_gt_zero_vmm_opmask, src_diff_vmm, vmm_ones_);
        vmulps(src_diff_vmm, src_diff_vmm, dst_diff_vmm);
        io_.at(diff_src_dt_)
                ->store(src_diff_vmm, data_ptr(DNNL_ARG_DIFF_SRC, offset),
                        tail);
        if (diff_src_block_tail_ && tail)
            prelu::apply_zero_padding(this, tail_size_, diff_src_dt_,
                    diff_src_block_tail_, reg_src_diff_, nullptr);
    }
}

template <typename Vmm>
void jit_uni_prelu_backward_kernel_t<Vmm>::accumulate_weights_diff(
        const Vmm &partial_sum_vmm, const Vmm &tmp_vmm,
        const Xbyak::Address &dst_addr, bool tail) {

    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked)) {
        uni_vaddps(
                weights_diff_acc_vmm_, weights_diff_acc_vmm_, partial_sum_vmm);
    } else if (bcast_ == prelu::bcast::per_oc_n_spatial_c) {
        if (std::is_same<Vmm, Xbyak::Zmm>::value || isa_ == avx2)
            uni_vaddps(partial_sum_vmm, partial_sum_vmm, dst_addr);
        else {
            uni_vmovups(tmp_vmm, dst_addr);
            uni_vaddps(partial_sum_vmm, partial_sum_vmm, tmp_vmm);
        }
        uni_vmovups(dst_addr, partial_sum_vmm);
    } else {
        io_.at(diff_wei_dt_)->store(partial_sum_vmm, dst_addr, tail);
        if (diff_wei_block_tail_ && tail)
            prelu::apply_zero_padding(this, tail_size_, diff_wei_dt_,
                    diff_wei_block_tail_, reg_weights_diff_, nullptr);
    }
}

template <typename Vmm>
const Xbyak::Operand &jit_uni_prelu_backward_kernel_t<Vmm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Vmm &weights_vmm, bool tail) {

    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;

    io_.at(wei_dt_)->load(src_addr, weights_vmm, tail);
    return weights_vmm;
}

static void reduce(jit_generator *host, const Xbyak::Xmm &src,
        const Xbyak::Xmm &helper, const cpu_isa_t &isa) {
    UNUSED(helper);
    if (isa == sse41) {
        host->haddps(src, src);
        host->haddps(src, src);
    } else {
        host->vhaddps(src, src, src);
        host->vhaddps(src, src, src);
    }
}

static void reduce(jit_generator *host, const Xbyak::Ymm &src,
        const Xbyak::Ymm &helper, const cpu_isa_t &isa) {
    const Xbyak::Xmm xmm_helper {helper.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    host->vextractf128(xmm_helper, src, 1);
    host->vaddps(xmm_src, xmm_src, xmm_helper);
    reduce(host, xmm_src, xmm_helper, isa);
}

static void reduce(jit_generator *host, const Xbyak::Zmm &src,
        const Xbyak::Zmm &helper, const cpu_isa_t &isa) {
    const Xbyak::Ymm ymm_helper {helper.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    host->vextractf64x4(ymm_helper, src, 1);
    host->vaddps(ymm_src, ymm_src, ymm_helper);
    reduce(host, ymm_src, ymm_helper, isa);
}

template <typename Vmm>
void jit_uni_prelu_backward_kernel_t<Vmm>::finalize() {
    if (bcast_ == prelu::bcast::per_oc_blocked)
        uni_vmovups(ptr[reg_weights_diff_], weights_diff_acc_vmm_);
    else if (bcast_ == prelu::bcast::per_oc_n_c_spatial) {
        reduce(this, weights_diff_acc_vmm_, weights_const_vmm_, isa_);
        uni_vmovss(ptr[reg_weights_diff_], weights_diff_acc_vmm_);
    }
}

template <typename Vmm>
std::map<data_type_t, io::io_saturation_conf_t>
jit_uni_prelu_backward_kernel_t<Vmm>::create_saturation_vmm_map() const {

    std::map<data_type_t, io::io_saturation_conf_t> saturation_map {};

    if (saturation_needed_diff_src_)
        saturation_map.emplace(diff_src_dt_,
                io::io_saturation_conf_t {vmm_zeros_.getIdx(),
                        saturation_ubound_diff_src_.getIdx(), reg_tmp_});

    if (saturation_needed_diff_weights_ && diff_src_dt_ != diff_wei_dt_)
        saturation_map.emplace(diff_wei_dt_,
                io::io_saturation_conf_t {vmm_zeros_.getIdx(),
                        saturation_ubound_diff_weights_.getIdx(), reg_tmp_});

    return saturation_map;
}

jit_prelu_backward_kernel_t *jit_prelu_backward_kernel_t::create(
        const cpu_prelu_bwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();

    const auto &src_dt = pd->src_md(0)->data_type;
    const auto &wei_dt = pd->weights_md(0)->data_type;
    const auto &diff_src_dt = pd->diff_src_md(0)->data_type;
    const auto &diff_dst_dt = pd->diff_dst_md(0)->data_type;
    const auto &diff_wei_dt = pd->diff_weights_md(0)->data_type;

    if (is_superset(isa, avx512_common))
        return new jit_uni_prelu_backward_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (is_superset(isa, avx)) {
        if (isa == avx
                && prelu::is_s8u8({src_dt, wei_dt, diff_src_dt, diff_dst_dt,
                        diff_wei_dt}))
            return new jit_uni_prelu_backward_kernel_t<Xbyak::Xmm>(pd, isa);
        else
            return new jit_uni_prelu_backward_kernel_t<Xbyak::Ymm>(pd, isa);
    } else if (isa == sse41)
        return new jit_uni_prelu_backward_kernel_t<Xbyak::Xmm>(pd, isa);

    return nullptr;
}

template class jit_uni_prelu_backward_kernel_t<Xbyak::Zmm>;
template class jit_uni_prelu_backward_kernel_t<Xbyak::Ymm>;
template class jit_uni_prelu_backward_kernel_t<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
