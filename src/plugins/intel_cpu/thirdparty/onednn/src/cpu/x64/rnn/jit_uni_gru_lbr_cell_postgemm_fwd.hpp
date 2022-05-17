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

#ifndef CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_GRU_LBR_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_lbr_cell_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_lbr_cell_postgemm_fwd)

    using injector_t = typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type;

    jit_uni_gru_lbr_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd) {}

    status_t init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        // we use rax for both constant tables and load correspondent label
        // into it when calling correspondent injector.
        sigmoid_injector_ = utils::make_unique<injector_t>(
                this, alg_kind::eltwise_logistic, 0.0f, 0.0f, 1.0f, true, rax);
        tanh_injector_ = utils::make_unique<injector_t>(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    std::unique_ptr<injector_t> sigmoid_injector_;
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    static constexpr size_t vlen = cpu_isa_traits<isa>::vlen;

    const size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias_ = vlen / (sizeof(float) / bias_dt_size_);
    const size_t hstate_dt_size = types::data_type_size(src_data_t);
    const size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    const size_t gate_dt_size = types::data_type_size(src_data_t);

    void generate() override {
        using namespace Xbyak;

        const auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;
        Label table_label;

        // Register map
        const Reg64 loop_cnt(r10); // loop counter
        const Reg64 table_reg(rbx); // table is used for data scale and shifts

        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        const Vmm G0(1), G1(2), G2(3), tmp1_vmm(5), tmp2_vmm(6);

        // constant table map
        const Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r11;
        const auto addr_states_tm1_l_reg = r12;
        const auto addr_scratch_cell_reg = rsi;
        const auto addr_ws_h_reg = rdi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_scratch_cell_reg, ptr[base_args + 16]);
        mov(addr_ws_h_reg, ptr[base_args + 24]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_states_tm1_l_reg = abi_param6;
        const auto addr_scratch_cell_reg = r11;
        const auto addr_ws_h_reg = r12;
        const auto base_args = get_stack_params_address();
        mov(addr_scratch_cell_reg, ptr[base_args]);
        mov(addr_ws_h_reg, ptr[base_args + 8]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        const auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_];
        };
        const auto sc_addr = [&](int i) {
            return ptr[addr_scratch_cell_reg + i * rnn_.dhc * scratch_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen);

        mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // Compute gate 0
            uni_vmovups(G0, sg_addr(0));
            to_float(tmp1_vmm, B_addr(0), rnn_.bias_dt, vlen);
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovups(tmp1_vmm, sc_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0.getIdx());
            // if training we write back the gates
            if (is_training) to_src(wg_addr(0), G0, src_data_t, vlen);

            // Compute gate 1
            uni_vmovups(G1, sg_addr(1));
            to_float(tmp1_vmm, B_addr(1), rnn_.bias_dt, vlen);
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovups(tmp1_vmm, sc_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G1.getIdx());
            // if training we write back the gates
            if (is_training) to_src(wg_addr(1), G1, src_data_t, vlen);

            // compute last gate
            const auto wh_b_addr = sc_addr(2);
            const auto ws_h_addr = ptr[addr_ws_h_reg];
            uni_vmovups(tmp1_vmm, wh_b_addr);
            to_float(tmp2_vmm, B_addr(3), rnn_.bias_dt, vlen);
            uni_vaddps(tmp1_vmm, tmp1_vmm, tmp2_vmm);
            if (is_training) to_src(ws_h_addr, tmp1_vmm, src_data_t, vlen);
            uni_vmovups(G2, sg_addr(2));
            to_float(tmp2_vmm, B_addr(2), rnn_.bias_dt, vlen);
            uni_vaddps(G2, G2, tmp2_vmm);
            uni_vfmadd231ps(G2, G1, tmp1_vmm);
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());
            // if training we write back the gates
            if (is_training) to_src(wg_addr(2), G2, src_data_t, vlen);

            // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
            uni_vmovups(tmp1_vmm, one_addr);
            uni_vsubps(tmp1_vmm, tmp1_vmm, G0);
            to_float(tmp2_vmm, ptr[addr_states_tm1_l_reg], src_data_t, vlen);
            uni_vmulps(G0, G0, tmp2_vmm);
            uni_vfmadd231ps(G0, tmp1_vmm, G2);

            // write back the result
            to_src(ptr[addr_states_t_l_reg], G0, src_data_t, vlen);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(vector_loop_inc_regs);
            to_src(ptr[addr_states_t_l_copy_reg], G0, src_data_t, vlen, true);

            // increment address pointers
            L(vector_loop_inc_regs);
            add(addr_scratch_gates_reg, vlen);
            add(addr_ws_h_reg, vlen_dst);
            add(addr_bias_reg, vlen_bias_);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_states_t_l_copy_reg, vlen_dst);
            add(addr_states_tm1_l_reg, vlen_dst);
            add(addr_scratch_cell_reg, vlen);
            if (is_training) add(addr_ws_gates_reg, vlen_dst);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            // remaping registers to Xmms
            const Xmm G0s(G0.getIdx()), G1s(G1.getIdx()), G2s(G2.getIdx());
            const Xmm tmp1s_vmm(tmp1_vmm.getIdx()),
                    tmp2s_vmm(tmp2_vmm.getIdx());

            // Compute gate 0
            uni_vmovss(G0s, sg_addr(0));
            to_float(tmp1s_vmm, B_addr(0), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G0s, G0s, tmp1s_vmm);
            uni_vaddss(G0s, G0s, sc_addr(0));
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0s.getIdx());
            // if training we write back the gates
            if (is_training)
                to_src(wg_addr(0), G0, src_data_t, scratch_dt_size);

            // Compute gate 1
            uni_vmovss(G1s, sg_addr(1));
            to_float(tmp1s_vmm, B_addr(1), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G1s, G1s, tmp1s_vmm);
            uni_vaddss(G1s, G1s, sc_addr(1));
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G1s.getIdx());
            // if training we write back the gates
            if (is_training)
                to_src(wg_addr(1), G1, src_data_t, scratch_dt_size);

            // compute last gate
            const auto wh_b_addr = sc_addr(2);
            const auto ws_h_addr = ptr[addr_ws_h_reg];
            uni_vmovss(tmp1s_vmm, wh_b_addr);
            to_float(tmp2s_vmm, B_addr(3), rnn_.bias_dt, sizeof(float));
            uni_vaddss(tmp1s_vmm, tmp1s_vmm, tmp2s_vmm);
            if (is_training)
                to_src(ws_h_addr, tmp1_vmm, src_data_t, scratch_dt_size);
            uni_vmovss(G2s, sg_addr(2));
            to_float(tmp2s_vmm, B_addr(2), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G2s, G2s, tmp2s_vmm);
            uni_vfmadd231ss(G2s, G1s, tmp1s_vmm);
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2s.getIdx());
            // if training we write back the gates
            if (is_training)
                to_src(wg_addr(2), G2, src_data_t, scratch_dt_size);

            // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
            uni_vmovss(tmp1s_vmm, one_addr);
            uni_vsubss(tmp1s_vmm, tmp1s_vmm, G0s);
            to_float(tmp2s_vmm, ptr[addr_states_tm1_l_reg], src_data_t,
                    scratch_dt_size);
            uni_vmulss(G0s, G0s, tmp2s_vmm);
            uni_vfmadd231ss(G0s, tmp1s_vmm, G2s);

            // write back the result
            to_src(ptr[addr_states_t_l_reg], G0, src_data_t, scratch_dt_size);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, rnn_.dhc * hstate_dt_size);
            jle(rem_loop_inc_regs);
            to_src(ptr[addr_states_t_l_copy_reg], G0, src_data_t,
                    scratch_dt_size, true);

            // increment address pointers
            L(rem_loop_inc_regs);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_ws_h_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size_);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_states_t_l_copy_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, hstate_dt_size);
            add(addr_scratch_cell_reg, scratch_dt_size);
            if (is_training) add(addr_ws_gates_reg, gate_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);
        init_table(vlen);

        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
    }
}; // namespace cpu

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
