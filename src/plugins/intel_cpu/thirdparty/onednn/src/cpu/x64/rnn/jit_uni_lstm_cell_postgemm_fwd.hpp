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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP

#include <memory>
#include "common/utils.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_fwd
    : public jit_uni_rnn_postgemm,
      public jit_uni_lstm_cell_postgemm_t<isa> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd)

    jit_uni_lstm_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd)
        , jit_uni_lstm_cell_postgemm_t<isa>(
                  this, 6 /*tmp_id_begin*/, static_cast<bool>(bf16_emu_)) {}

    ~jit_uni_lstm_cell_postgemm_fwd() = default;

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
    using injector_t = typename jit_uni_lstm_cell_postgemm_t<isa>::injector_t;
    using Vmm = typename jit_uni_lstm_cell_postgemm_t<isa>::Vmm;

    std::unique_ptr<injector_t> sigmoid_injector_;
    std::unique_ptr<injector_t> tanh_injector_;

    // register size in bytes
    static constexpr size_t vlen_ = cpu_isa_traits<isa>::vlen;
    static constexpr size_t qscale_dt_size = sizeof(float);
    static constexpr size_t weights_peephole_dt_size_ = sizeof(float);
    const size_t vlen_dst_
            = vlen_ / (sizeof(float) / types::data_type_size(src_data_t));
    const size_t vlen_bias_ = vlen_ / (sizeof(float) / bias_dt_size_);
    const size_t vlen_c_states_ = vlen_ / (sizeof(float) / cstate_dt_size_);
    const size_t hstate_dt_size_ = types::data_type_size(src_data_t);
    const size_t gate_dt_size_ = types::data_type_size(src_data_t);
    const size_t scratch_dt_size_ = types::data_type_size(scratch_data_t);

    void generate() override {
        using namespace Xbyak;

        const auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *const weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;

        // Register map
        const Reg64 loop_cnt(rbx); // loop counter

        // We start code generations here
        preamble();

        const Reg64 n_step_reg(rbp);

        // extract addresses passed as parameter
        const auto addr_ws_gates_reg = abi_param1;
        const auto addr_scratch_gates_reg = abi_param2;
        const auto addr_weights_peephole_reg = r11;
        const auto addr_bias_reg = abi_param3;
        const auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        const auto addr_states_t_l_copy_reg = r10;
        const auto addr_c_states_tm1_l_reg = rdi;
        const auto addr_c_states_t_l_reg = rsi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        const auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 16]);
        mov(addr_weights_peephole_reg, ptr[base_args + 24]);
        mov(n_step_reg, ptr[base_args + 40]);
#else
        const auto addr_states_t_l_copy_reg = abi_param5;
        const auto addr_c_states_tm1_l_reg = abi_param6;
        const auto addr_c_states_t_l_reg = r10;
        const auto base_args = get_stack_params_address();
        mov(addr_c_states_t_l_reg, ptr[base_args]);
        mov(addr_weights_peephole_reg, ptr[base_args + 8]);
        mov(n_step_reg, ptr[base_args + 24]);
#endif

        // helper lambda to address the gates and biases
        const auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg
                    + i * rnn_.dhc * scratch_dt_size_];
        };
        const auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size_];
        };
        const auto weights_peephole_addr = [&](int i) {
            return ptr[addr_weights_peephole_reg
                    + i * rnn_.dhc * weights_peephole_dt_size_];
        };
        const auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size_];
        };

        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen_);

        sigmoid_injector_->load_table_addr();
        tanh_injector_->load_table_addr();
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(loop_cnt, n_step_reg);
        else
            mov(loop_cnt, rnn_.dhc * scratch_dt_size_);
        cmp(loop_cnt, vlen_);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L_aligned(vector_loop_start_label, 64);
        {
            const Vmm G0(1), G1(2), G2(4), G3(3), tmp_c_states(5);
            // load G0 G1 G2 G3
            uni_vmovups(G0, sg_addr(0));
            uni_vmovups(G1, sg_addr(1));
            uni_vmovups(G2, sg_addr(2));
            uni_vmovups(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed, add bias
            deq_w(src_data_t, G0, this->get_next_tmp_vmm(),
                    this->get_next_tmp_vmm(), 0 * rnn_.dhc, mask, true);
            const auto bias_g0_vmm = this->get_next_tmp_vmm();
            to_float(bias_g0_vmm, B_addr(0), rnn_.bias_dt, vlen_);
            this->uni_vaddps(G0, G0, bias_g0_vmm);

            deq_w(src_data_t, G1, this->get_next_tmp_vmm(),
                    this->get_next_tmp_vmm(), 1 * rnn_.dhc, mask, true);
            const auto bias_g1_vmm = this->get_next_tmp_vmm();
            to_float(bias_g1_vmm, B_addr(1), rnn_.bias_dt, vlen_);
            this->uni_vaddps(G1, G1, bias_g1_vmm);

            deq_w(src_data_t, G2, this->get_next_tmp_vmm(),
                    this->get_next_tmp_vmm(), 2 * rnn_.dhc, mask, true);
            const auto bias_g2_vmm = this->get_next_tmp_vmm();
            to_float(bias_g2_vmm, B_addr(2), rnn_.bias_dt, vlen_);
            this->uni_vaddps(G2, G2, bias_g2_vmm);

            deq_w(src_data_t, G3, this->get_next_tmp_vmm(),
                    this->get_next_tmp_vmm(), 3 * rnn_.dhc, mask, true);
            const auto bias_g3_vmm = this->get_next_tmp_vmm();
            to_float(bias_g3_vmm, B_addr(3), rnn_.bias_dt, vlen_);
            this->uni_vaddps(G3, G3, bias_g3_vmm);

            to_float(tmp_c_states, ptr[addr_c_states_tm1_l_reg],
                    rnn_.src_iter_c_dt, vlen_);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ps_rhs_op_mem(
                        G0, tmp_c_states, weights_peephole_addr(0));
                this->vfmadd231ps_rhs_op_mem(
                        G1, tmp_c_states, weights_peephole_addr(1));
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            const auto sigmoid_range_begin = G0.getIdx();
            const auto sigmoid_range_end
                    = (rnn_.is_lstm_peephole ? G1.getIdx() : G3.getIdx()) + 1;
            sigmoid_injector_->compute_vector_range(
                    sigmoid_range_begin, sigmoid_range_end);

            if (is_training) {
                to_src(wg_addr(0), G0, src_data_t, vlen_);
                to_src(wg_addr(1), G1, src_data_t, vlen_);
                if (!rnn_.is_lstm_peephole)
                    to_src(wg_addr(3), G3, src_data_t, vlen_);
            }
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());

            if (is_training) { to_src(wg_addr(2), G2, src_data_t, vlen_); }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmulps(tmp_c_states, tmp_c_states, G1);
            const auto tmp_g0 = this->vmm_backup(G0);
            uni_vfmadd231ps(tmp_c_states, tmp_g0, G2);
            to_src(ptr[addr_c_states_t_l_reg], tmp_c_states, rnn_.dst_iter_c_dt,
                    vlen_);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ps_rhs_op_mem(
                        G3, tmp_c_states, weights_peephole_addr(2));
                sigmoid_injector_->load_table_addr();
                sigmoid_injector_->compute_vector(G3.getIdx());

                // if training we write back the gates
                if (is_training) { to_src(wg_addr(3), G3, src_data_t, vlen_); }
            }

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp_c_states.getIdx());
            uni_vmulps(tmp_c_states, tmp_c_states, G3);

            // downconvert and write back the state
            to_src(ptr[addr_states_t_l_reg], tmp_c_states, src_data_t, vlen_);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, 0);
            je(vector_loop_inc_regs);
            to_src(ptr[addr_states_t_l_copy_reg], tmp_c_states, src_data_t,
                    vlen_, true);
            add(addr_states_t_l_copy_reg, vlen_dst_);

            // increment address pointers
            L_aligned(vector_loop_inc_regs);
            add(addr_scratch_gates_reg, vlen_);
            if (rnn_.is_lstm_peephole) add(addr_weights_peephole_reg, vlen_);
            add(addr_bias_reg, vlen_bias_);
            add(addr_states_t_l_reg, vlen_dst_);
            add(addr_c_states_tm1_l_reg, vlen_c_states_);
            add(addr_c_states_t_l_reg, vlen_c_states_);
            if (is_training) add(addr_ws_gates_reg, vlen_dst_);
            inc_regs(mask, vlen_);

            // increment loop counter
            sub(loop_cnt, vlen_);
            cmp(loop_cnt, vlen_);
            jge(vector_loop_start_label);
        }
        L_aligned(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        this->reset_vmm_cnt();

        L_aligned(rem_loop_start_label, 64);
        {
            const Xmm G0(1), G1(2), G2(4), G3(3), tmp_c_states(5);
            // load G0 G1 G2 G3
            uni_vmovss(G0, sg_addr(0));
            uni_vmovss(G1, sg_addr(1));
            uni_vmovss(G2, sg_addr(2));
            uni_vmovss(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed
            deq_w(src_data_t, G0, this->get_next_tmp_xmm(),
                    this->get_next_tmp_xmm(), 0 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G1, this->get_next_tmp_xmm(),
                    this->get_next_tmp_xmm(), 1 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G2, this->get_next_tmp_xmm(),
                    this->get_next_tmp_xmm(), 2 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G3, this->get_next_tmp_xmm(),
                    this->get_next_tmp_xmm(), 3 * rnn_.dhc, mask, false);

            // add biases
            const auto bias_g0_xmm = this->get_next_tmp_xmm();
            to_float(bias_g0_xmm, B_addr(0), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G0, G0, bias_g0_xmm);
            const auto bias_g1_xmm = this->get_next_tmp_xmm();
            to_float(bias_g1_xmm, B_addr(1), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G1, G1, bias_g1_xmm);
            const auto bias_g2_xmm = this->get_next_tmp_xmm();
            to_float(bias_g2_xmm, B_addr(2), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G2, G2, bias_g2_xmm);
            const auto bias_g3_xmm = this->get_next_tmp_xmm();
            to_float(bias_g3_xmm, B_addr(3), rnn_.bias_dt, sizeof(float));
            uni_vaddss(G3, G3, bias_g3_xmm);

            to_float(tmp_c_states, ptr[addr_c_states_tm1_l_reg],
                    rnn_.src_iter_c_dt, sizeof(float));
            // add peephole
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ss_rhs_op_mem(
                        G0, tmp_c_states, weights_peephole_addr(0));
                this->vfmadd231ss_rhs_op_mem(
                        G1, tmp_c_states, weights_peephole_addr(1));
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            const auto sigmoid_range_begin = G0.getIdx();
            const auto sigmoid_range_end
                    = (rnn_.is_lstm_peephole ? G1.getIdx() : G3.getIdx()) + 1;
            sigmoid_injector_->compute_vector_range(
                    sigmoid_range_begin, sigmoid_range_end);

            if (is_training) {
                to_src(wg_addr(0), G0, src_data_t, scratch_dt_size_);
                to_src(wg_addr(1), G1, src_data_t, scratch_dt_size_);
                if (!rnn_.is_lstm_peephole)
                    to_src(wg_addr(3), G3, src_data_t, scratch_dt_size_);
            }

            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());
            if (is_training)
                to_src(wg_addr(2), G2, src_data_t, scratch_dt_size_);

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmulss(tmp_c_states, tmp_c_states, G1);
            const auto tmp_g0 = this->xmm_backup(G0);
            uni_vfmadd231ss(tmp_c_states, tmp_g0, G2);
            to_src(ptr[addr_c_states_t_l_reg], tmp_c_states, rnn_.dst_iter_c_dt,
                    sizeof(float));
            // add peephole
            if (rnn_.is_lstm_peephole) {
                this->vfmadd231ss_rhs_op_mem(
                        G3, tmp_c_states, weights_peephole_addr(2));
                sigmoid_injector_->load_table_addr();
                sigmoid_injector_->compute_vector(G3.getIdx());
                // if training we write back the gates
                if (is_training)
                    to_src(wg_addr(3), G3, src_data_t, scratch_dt_size_);
            }

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp_c_states.getIdx());
            uni_vmulss(tmp_c_states, tmp_c_states, G3);

            // downconcvert/quantize and write back the state
            to_src(ptr[addr_states_t_l_reg], tmp_c_states, src_data_t,
                    scratch_dt_size_);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, 0);
            je(rem_loop_inc_regs);
            to_src(ptr[addr_states_t_l_copy_reg], tmp_c_states, src_data_t,
                    scratch_dt_size_, true);
            add(addr_states_t_l_copy_reg, hstate_dt_size_);

            // increment address pointers
            L_aligned(rem_loop_inc_regs);
            add(addr_scratch_gates_reg, scratch_dt_size_);
            if (rnn_.is_lstm_peephole)
                add(addr_weights_peephole_reg, weights_peephole_dt_size_);
            add(addr_bias_reg, bias_dt_size_);
            add(addr_states_t_l_reg, hstate_dt_size_);
            add(addr_c_states_tm1_l_reg, cstate_dt_size_);
            add(addr_c_states_t_l_reg, cstate_dt_size_);
            if (is_training) add(addr_ws_gates_reg, gate_dt_size_);
            inc_regs(mask, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size_);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L_aligned(rem_loop_end_label);

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);

        init_table(vlen_);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
