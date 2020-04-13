/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_JIT_GRU_LBR_CELL_POSTGEMM
#define CPU_JIT_GRU_LBR_CELL_POSTGEMM

#include "jit_uni_rnn_common_postgemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t>
struct jit_uni_gru_lbr_cell_postgemm_fwd: public jit_uni_rnn_postgemm
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_lbr_cell_postgemm_fwd)

    typedef typename utils::conditional<src_data_t == data_type::u8, int32_t,
            float>::type acc_data_t;
    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_gru_lbr_cell_postgemm_fwd(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
    : jit_uni_rnn_postgemm(rnn, pd){}

    ~jit_uni_gru_lbr_cell_postgemm_fwd(){
        delete sigmoid_injector_;
        delete tanh_injector_;
    }

    void init() override {
        // we use rax for both constant tables as they use the same table
        sigmoid_injector_ = new injector_t(this,
                alg_kind::eltwise_logistic, 0.0f, 0.0f, true, rax);
        tanh_injector_ = new injector_t(this,
                alg_kind::eltwise_tanh, 0.0f, 0.0f, true, rax);
        generate();
        kernel_ = (kernel_t) this->getCode();
    }

protected:
    injector_t *sigmoid_injector_;
    injector_t *tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_dst = (src_data_t == data_type::u8) ? vlen/4 : vlen;
    size_t hstate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint8_t) : sizeof(float);
    size_t gate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint32_t) : sizeof(float);
    size_t bias_dt_size = sizeof(float);

    void generate() {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 loop_cnt(r11);  // loop counter
        Reg64 table_reg(rbx); // table is used for data scale and shifts

        // We skip vmm0 as it can be used by the injector for masks on sse4.2
        Vmm G0(1), G1(2), G2(3), tmp1_vmm(5), tmp2_vmm(6);

        // constant table map
        Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;
        auto addr_states_tm1_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_scratch_cell_reg = r10;
        auto addr_ws_h_reg = rdi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        mov(addr_scratch_cell_reg, ptr[rsp + get_size_of_abi_save_regs() + 40]);
        mov(addr_ws_h_reg, ptr[rsp + get_size_of_abi_save_regs() + 48]);
#else
        auto addr_scratch_cell_reg = abi_param5;
        auto addr_ws_h_reg = abi_param6;
#endif

        // helper lambda to address the gates and biases
        auto G_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dic * gate_dt_size];
        };
        auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dic * bias_dt_size];
        };
        auto S_addr = [&](int i) {
            return ptr[addr_scratch_cell_reg + i * rnn_.dic * gate_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);

        // both sigmoid and tanh use the same table so load address just once in rax
        sigmoid_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dic * gate_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // Compute gate 0
            uni_vmovups(G0, G_addr(0));
            uni_vmovups(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovups(tmp1_vmm, S_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            sigmoid_injector_->compute_vector(G0.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovups(G_addr(0), G0);

            // Compute gate 1
            uni_vmovups(G1, G_addr(1));
            uni_vmovups(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovups(tmp1_vmm, S_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            sigmoid_injector_->compute_vector(G1.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovups(G_addr(1), G1);

            // compute last gate
            auto wh_b_addr = S_addr(2);
            auto ws_h_addr = ptr[addr_ws_h_reg];
            uni_vmovups(tmp1_vmm, wh_b_addr);
            uni_vmovups(tmp2_vmm, B_addr(3));
            uni_vaddps(tmp1_vmm, tmp1_vmm, tmp2_vmm);
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovups(ws_h_addr, tmp1_vmm);
            uni_vmovups(G2, G_addr(2));
            uni_vmovups(tmp2_vmm, B_addr(2));
            uni_vaddps(G2, G2, tmp2_vmm);
            uni_vfmadd231ps(G2, G1, tmp1_vmm);
            tanh_injector_->compute_vector(G2.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovups(G_addr(2), G2);

            // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
            uni_vmovups(tmp1_vmm, one_addr);
            uni_vsubps(tmp1_vmm, tmp1_vmm, G0);
            uni_vmovups(tmp2_vmm, ptr[addr_states_tm1_l_reg]);
            uni_vmulps(G0, G0, tmp2_vmm);
            uni_vfmadd231ps(G0, tmp1_vmm, G2);

            // write back the result
            uni_vmovups(ptr[addr_states_t_l_reg], G0);

            // increment address pointers
            add(addr_ws_gates_reg, vlen);
            add(addr_ws_h_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_states_tm1_l_reg, vlen_dst);
            add(addr_scratch_cell_reg, vlen_dst);

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
            Xmm G0s(G0.getIdx()), G1s(G1.getIdx()), G2s(G2.getIdx());
            Xmm tmp1s_vmm(tmp1_vmm.getIdx());

            // Compute gate 0
            uni_vmovss(G0s, G_addr(0));
            uni_vaddss(G0s, G0s, B_addr(0));
            uni_vaddss(G0s, G0s, S_addr(0));
            sigmoid_injector_->compute_vector(G0s.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(G_addr(0), G0s);

            // Compute gate 1
            uni_vmovss(G1s, G_addr(1));
            uni_vaddss(G1s, G1s, B_addr(1));
            uni_vaddss(G1s, G1s, S_addr(1));
            sigmoid_injector_->compute_vector(G1s.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(G_addr(1), G1s);

            // compute last gate
            auto wh_b_addr = S_addr(2);
            auto ws_h_addr = ptr[addr_ws_h_reg];
            uni_vmovss(tmp1s_vmm, wh_b_addr);
            uni_vaddss(tmp1s_vmm, tmp1s_vmm, B_addr(3));
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(ws_h_addr, tmp1s_vmm);
            uni_vmovss(G2s, G_addr(2));
            uni_vaddss(G2s, G2s, B_addr(2));
            uni_vfmadd231ss(G2s, G1s, tmp1s_vmm);
            tanh_injector_->compute_vector(G2s.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(G_addr(2), G2s);

            // states_t_l = states_tm1_l * G0 + (1 - G0) * G2
            uni_vmovss(tmp1s_vmm, one_addr);
            uni_vsubss(tmp1s_vmm, tmp1_vmm, G0s);
            uni_vmulss(G0s, G0s, ptr[addr_states_tm1_l_reg]);
            uni_vfmadd231ss(G0s, tmp1s_vmm, G2s);

            // write back the result
	    uni_vmovss(ptr[addr_states_t_l_reg], G0s);

            // increment address pointers
            add(addr_ws_gates_reg, gate_dt_size);
            add(addr_ws_h_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, hstate_dt_size);
            add(addr_scratch_cell_reg, gate_dt_size);

            // increment loop counter
            sub(loop_cnt, gate_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);

        }
        L(rem_loop_end_label);

        postamble();

        // Again, only one table is needed and shared between sigmoid and tanh
        sigmoid_injector_->prepare_table(false);
        tanh_injector_->prepare_table(true);

        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++) dd(float2int(1.0f));
        }
    }
}; // namespace cpu

template struct jit_uni_gru_lbr_cell_postgemm_fwd<sse42, data_type::f32>;
template struct jit_uni_gru_lbr_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_gru_lbr_cell_postgemm_fwd<avx512_core, data_type::f32>;

}
}
}

#endif
