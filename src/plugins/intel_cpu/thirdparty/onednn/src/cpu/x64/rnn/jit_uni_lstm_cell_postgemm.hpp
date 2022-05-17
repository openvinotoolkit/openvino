/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_HPP

#include "common/utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_lstm_cell_postgemm_t {
    jit_uni_lstm_cell_postgemm_t(
            jit_generator *host, int tmp_id_begin, bool use_bf16_emu)
        : host_(host)
        , tmp_id_begin_(tmp_id_begin)
        , current_tmp_id_(tmp_id_begin)
        , tmp_id_end_(cpu_isa_traits<isa>::n_vregs
                  - (is_superset(isa, avx512_common) && use_bf16_emu ? 4 : 0)) {
    }

protected:
    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type;
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const size_t vlen_ = cpu_isa_traits<isa>::vlen;

    Vmm get_next_tmp_vmm() {
        const Vmm vmm {current_tmp_id_++};

        if (current_tmp_id_ == tmp_id_end_) reset_vmm_cnt();

        return vmm;
    }

    void reset_vmm_cnt() { current_tmp_id_ = tmp_id_begin_; }

    Xbyak::Xmm get_next_tmp_xmm() {
        return Xbyak::Xmm(get_next_tmp_vmm().getIdx());
    }

    Vmm vmm_backup(const Vmm &vmm) {
        auto tmp_vmm = vmm;
        if (!this->avx2_available_) {
            tmp_vmm = this->get_next_tmp_vmm();
            host_->uni_vmovups(tmp_vmm, vmm);
        }
        return tmp_vmm;
    };

    Xbyak::Xmm xmm_backup(const Xbyak::Xmm &xmm) {
        auto tmp_xmm = xmm;
        if (!this->avx2_available_) {
            tmp_xmm = this->get_next_tmp_xmm();
            host_->uni_vmovss(tmp_xmm, xmm);
        }
        return tmp_xmm;
    };

    void vaddps_rhs_op_mem(
            const Vmm &dst, const Vmm &lhs, const Xbyak::Address &rhs_addr) {

        if (avx2_available_)
            host_->uni_vaddps(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_vmm();
            host_->uni_vmovups(rhs, rhs_addr);
            host_->uni_vaddps(dst, lhs, rhs);
        }
    }

    void vfmadd231ps_rhs_op_mem(
            const Vmm &dst, const Vmm &lhs, const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            host_->uni_vfmadd231ps(dst, lhs, rhs_addr);
        else {
            const auto tmp = get_next_tmp_vmm();
            host_->uni_vmovups(tmp, rhs_addr);
            const auto &rhs = lhs;
            host_->uni_vfmadd231ps(dst, tmp, rhs);
        }
    }

    void vmulps_rhs_op_mem(
            const Vmm &dst, const Vmm &lhs, const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            host_->uni_vmulps(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_vmm();
            host_->uni_vmovups(rhs, rhs_addr);
            host_->uni_vmulps(dst, lhs, rhs);
        }
    }

    void vaddss_rhs_op_mem(const Xbyak::Xmm &dst, const Xbyak::Xmm &lhs,
            const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            host_->uni_vaddss(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_xmm();
            host_->uni_vmovss(rhs, rhs_addr);
            host_->uni_vaddss(dst, lhs, rhs);
        }
    }

    void vfmadd231ss_rhs_op_mem(const Xbyak::Xmm &dst, const Xbyak::Xmm &lhs,
            const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            host_->uni_vfmadd231ss(dst, lhs, rhs_addr);
        else {
            const auto tmp = get_next_tmp_xmm();
            host_->uni_vmovss(tmp, rhs_addr);
            const auto &rhs = lhs;
            host_->uni_vfmadd231ss(dst, tmp, rhs);
        }
    }

    void vmulss_rhs_op_mem(const Xbyak::Xmm &dst, const Xbyak::Xmm &lhs,
            const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            host_->uni_vmulss(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_xmm();
            host_->uni_vmovss(rhs, rhs_addr);
            host_->uni_vmulss(dst, lhs, rhs);
        }
    }

protected:
    const bool avx2_available_ = is_superset(isa, avx2);

private:
    jit_generator *host_;
    const int tmp_id_begin_;
    int current_tmp_id_;
    const int tmp_id_end_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
