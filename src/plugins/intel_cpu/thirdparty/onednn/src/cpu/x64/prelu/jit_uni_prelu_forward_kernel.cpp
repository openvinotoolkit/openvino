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
#include <cstddef>

#include "cpu/x64/prelu/jit_uni_prelu_forward_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_prelu_forward_kernel_t::jit_prelu_forward_kernel_t(
        const cpu_prelu_fwd_pd_t *pd, const cpu_isa_t &isa, const int vlen,
        const size_t number_vmm_single_compute)
    : jit_prelu_base_kernel_t(isa, vlen,
            prelu::get_bcast_type(memory_desc_wrapper(pd->src_md(0)),
                    memory_desc_wrapper(pd->weights_md(0))),
            memory_desc_wrapper(pd->src_md(0)), number_vmm_single_compute)
    , src_dt_(pd->src_md(0)->data_type)
    , wei_dt_(pd->weights_md(0)->data_type)
    , dst_dt_(pd->dst_md(0)->data_type)
    , dst_tail_block_(prelu::get_block_tail_size(pd->dst_md(0)))
    , pd_(pd) {}

#define PARAM_OFF(x) offsetof(call_params_t, x)

void jit_prelu_forward_kernel_t::load_kernel_call_params() {
    mov(reg_src_, ptr[abi_param1 + PARAM_OFF(src)]);
    mov(reg_weights_, ptr[abi_param1 + PARAM_OFF(weights)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst)]);
    mov(reg_data_size_, ptr[abi_param1 + PARAM_OFF(compute_data_size)]);
}

#undef PARAM_OFF

Xbyak::Address jit_prelu_forward_kernel_t::data_ptr(int arg_num, size_t offt) {

    const auto get_addr
            = [&](const Xbyak::Reg64 &reg_base, const data_type_t dt) {
                  const auto dt_size = types::data_type_size(dt);
                  return ptr[reg_base + reg_offset_ * dt_size + offt * dt_size];
              };

    switch (arg_num) {
        case DNNL_ARG_SRC: return get_addr(reg_src_, src_dt_);
        case DNNL_ARG_WEIGHTS: return get_addr(reg_weights_, wei_dt_);
        case DNNL_ARG_DST: return get_addr(reg_dst_, dst_dt_);
        default: assert(!"unsupported arg_num"); break;
    }
    return Xbyak::Address(0);
}

bool jit_prelu_forward_kernel_t::any_tensor_bf16() const {
    return utils::one_of(data_type::bf16, src_dt_, wei_dt_, dst_dt_);
}

template <typename Vmm>
jit_uni_prelu_forward_kernel_t<Vmm>::jit_uni_prelu_forward_kernel_t(
        const cpu_prelu_fwd_pd_t *pd, const cpu_isa_t &isa)
    : jit_prelu_forward_kernel_t(pd, isa, prelu::vmm_traits_t<Vmm>::vlen,
            (utils::one_of(isa, sse41, avx)
                    || pd->src_md(0)->data_type != data_type::f32)
                    ? 4u
                    : 3u)
    , saturation_needed_(utils::one_of(
              dst_dt_, data_type::u8, data_type::s8, data_type::s32))
    , vmm_zeros_(reserve_vmm())
    , dst_saturate_ubound_(saturation_needed_ ? reserve_vmm() : 0)
    , tail_vmm_mask_(
              tail_size_ && utils::one_of(isa, avx, avx2) ? reserve_vmm() : 0)
    , weights_const_vmm_(utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                                 prelu::bcast::per_oc_blocked)
                      ? reserve_vmm()
                      : 0)
    , io_(this, isa, {src_dt_, wei_dt_, dst_dt_}, {},
              io::io_tail_conf_t {simd_w_, tail_size_, tail_opmask_,
                      tail_vmm_mask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {}, create_saturation_vmm_map()) {}

template <typename Vmm>
jit_uni_prelu_forward_kernel_t<Vmm>::~jit_uni_prelu_forward_kernel_t()
        = default;

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::prepare_kernel_const_vars() {
    uni_vxorps(vmm_zeros_, vmm_zeros_, vmm_zeros_);

    io_.init_bf16();
    if (saturation_needed_) io_.init_saturate_f32({dst_dt_});
    if (tail_size_) io_.prepare_tail_mask();
    if (bcast_ == prelu::bcast::per_oc_n_c_spatial)
        io_.at(wei_dt_)->broadcast(ptr[reg_weights_], weights_const_vmm_);
    else if (bcast_ == prelu::bcast::per_oc_blocked)
        io_.at(wei_dt_)->load(
                ptr[reg_weights_], weights_const_vmm_, false /*tail*/);
}

template <typename Vmm>
std::map<data_type_t, io::io_saturation_conf_t>
jit_uni_prelu_forward_kernel_t<Vmm>::create_saturation_vmm_map() const {

    std::map<data_type_t, io::io_saturation_conf_t> saturation_map {};

    if (saturation_needed_) {
        saturation_map.emplace(dst_dt_,
                io::io_saturation_conf_t {vmm_zeros_.getIdx(),
                        dst_saturate_ubound_.getIdx(), reg_tmp_});
    }

    return saturation_map;
}

template <>
bool jit_uni_prelu_forward_kernel_t<
        Xbyak::Zmm>::can_load_wei_from_addr_directly(bool tail) const noexcept {
    return wei_dt_ == data_type::f32
            && !utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                    prelu::bcast::per_oc_blocked);
}

template <>
bool jit_uni_prelu_forward_kernel_t<
        Xbyak::Ymm>::can_load_wei_from_addr_directly(bool tail) const noexcept {
    return wei_dt_ == data_type::f32 && is_superset(isa_, avx2) && !tail
            && !utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                    prelu::bcast::per_oc_blocked);
}

template <>
bool jit_uni_prelu_forward_kernel_t<
        Xbyak::Xmm>::can_load_wei_from_addr_directly(bool tail) const noexcept {
    return false;
}

template <>
Xbyak::Zmm jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Xbyak::Zmm &weights_vmm,
        bool tail) {
    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;

    io_.at(wei_dt_)->load(src_addr, weights_vmm, tail);
    return weights_vmm;
}

template <>
Xbyak::Ymm jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Xbyak::Ymm &weights_vmm,
        bool tail) {
    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;

    io_.at(wei_dt_)->load(src_addr, weights_vmm, tail);
    return weights_vmm;
}

template <>
Xbyak::Xmm jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>::get_or_load_weights(
        const Xbyak::Address &src_addr, const Xbyak::Xmm &weights_vmm,
        bool tail) {

    if (utils::one_of(bcast_, prelu::bcast::per_oc_n_c_spatial,
                prelu::bcast::per_oc_blocked))
        return weights_const_vmm_;

    io_.at(wei_dt_)->load(src_addr, weights_vmm, tail);
    return weights_vmm;
}

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::uni_vfmadd132ps(
        const Vmm &x1, const Vmm &x2, const Xbyak::Operand &op, bool tail) {
    uni_vfmadd132ps(x1, x2, op);
}

template <>
void jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>::uni_vfmadd132ps(
        const Xbyak::Zmm &x1, const Xbyak::Zmm &x2, const Xbyak::Operand &op,
        bool tail) {

    if (op.isMEM()) {
        const Xbyak::Zmm dst = tail ? (x1 | tail_opmask_) : x1;
        // workaround for DataParallelC++ compiler issue converting mem to ZMM
        const Xbyak::Address addr
                = reinterpret_cast<const Xbyak::Address &>(op);
        vfmadd132ps(dst, x2, addr);
    } else {
        vfmadd132ps(x1, x2, op);
    }
}

template <typename Vmm>
void jit_uni_prelu_forward_kernel_t<Vmm>::compute_dst(
        size_t unrolling_factor, bool tail) {
    static constexpr size_t max_idx = 0;
    static constexpr size_t min_idx = 1;
    static constexpr size_t src_idx = 2;
    static constexpr size_t weights_idx = 3;

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {
        const Vmm max_vmm {get_compute_vmm(max_idx, unroll_group)};
        const Vmm min_vmm {get_compute_vmm(min_idx, unroll_group)};
        const Vmm src_vmm {get_compute_vmm(src_idx, unroll_group)};
        const Vmm weights_vmm {get_compute_vmm(weights_idx, unroll_group)};

        const auto offset = unroll_group * simd_w_;
        io_.at(src_dt_)->load(data_ptr(DNNL_ARG_SRC, offset), src_vmm, tail);
        uni_vmaxps(max_vmm, vmm_zeros_, src_vmm);
        uni_vminps(min_vmm, vmm_zeros_, src_vmm);
        const auto &dst_vmm = min_vmm;

        const Xbyak::Address weights_addr = data_ptr(DNNL_ARG_WEIGHTS, offset);
        if (can_load_wei_from_addr_directly(tail)) {
            uni_vfmadd132ps(dst_vmm, max_vmm, weights_addr, tail);
        } else {
            const Vmm weights_operand
                    = get_or_load_weights(weights_addr, weights_vmm, tail);
            uni_vfmadd132ps(dst_vmm, max_vmm, weights_operand, tail);
        }

        io_.at(dst_dt_)->store(dst_vmm, data_ptr(DNNL_ARG_DST, offset), tail);
        if (dst_tail_block_ && tail)
            prelu::apply_zero_padding(this, tail_size_, dst_dt_,
                    dst_tail_block_, reg_dst_, &reg_offset_);
    }
}

jit_prelu_forward_kernel_t *jit_prelu_forward_kernel_t::create(
        const cpu_prelu_fwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();
    const auto &src_dt = pd->src_md(0)->data_type;
    const auto &wei_dt = pd->weights_md(0)->data_type;
    const auto &dst_dt = pd->dst_md(0)->data_type;

    if (is_superset(isa, avx512_common))
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (is_superset(isa, avx))
        if (isa == avx && prelu::is_s8u8({src_dt, wei_dt, dst_dt}))
            return new jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>(pd, isa);
        else
            return new jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>(pd, isa);
    else if (isa == sse41)
        return new jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>(pd, isa);

    return nullptr;
}

template class jit_uni_prelu_forward_kernel_t<Xbyak::Zmm>;
template class jit_uni_prelu_forward_kernel_t<Xbyak::Ymm>;
template class jit_uni_prelu_forward_kernel_t<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
