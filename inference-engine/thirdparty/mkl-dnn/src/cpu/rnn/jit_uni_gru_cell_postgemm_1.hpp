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

#ifndef CPU_JIT_GRU_CELL_POSTGEMM_PART1
#define CPU_JIT_GRU_CELL_POSTGEMM_PART1

#include "jit_uni_rnn_common_postgemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t>
struct jit_uni_gru_cell_postgemm_part1_fwd: public jit_uni_rnn_postgemm
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_cell_postgemm_part1_fwd)

    typedef typename utils::conditional<src_data_t == data_type::u8, int32_t,
            float>::type acc_data_t;
    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_gru_cell_postgemm_part1_fwd(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
    : jit_uni_rnn_postgemm(rnn, pd){}

    ~jit_uni_gru_cell_postgemm_part1_fwd(){
        delete sigmoid_injector_;
    }

    void init() override {
        // we use rax for both constant tables as they use the same table
        sigmoid_injector_ = new injector_t(this,
                alg_kind::eltwise_logistic, 0.0f, 0.0f, true, rax);
        generate();
        kernel_ = (kernel_t) this->getCode();
    }

protected:
    injector_t *sigmoid_injector_;

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
        Vmm G0(1), G1(2), tmp1_vmm(3);

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;
        auto addr_states_tm1_l_reg = abi_param4;

        // helper lambda to address the gates and biases
        auto G_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dic * gate_dt_size];
        };
        auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dic * bias_dt_size];
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
            // Compute gate 0: G0 = sigmoid(G0 + b0)
            uni_vmovups(G0, G_addr(0));
            uni_vmovups(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            sigmoid_injector_->compute_vector(G0.getIdx());
            // we store it for use in postgemm_part2
            uni_vmovups(G_addr(0), G0);

            // Compute gate 1:  G1 = sigmoid(G1 + b1)
            uni_vmovups(G1, G_addr(1));
            uni_vmovups(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            sigmoid_injector_->compute_vector(G1.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovups(G_addr(1), G1);

            // states_t_l = states_tm1_l * G1
            uni_vmovups(tmp1_vmm, ptr[addr_states_tm1_l_reg]);
            uni_vmulps(G1, G1, tmp1_vmm);
            uni_vmovups(ptr[addr_states_t_l_reg], G1);

            // increment address pointers
            add(addr_ws_gates_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_states_tm1_l_reg, vlen_dst);
 
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
            Xmm G0s(G0.getIdx()), G1s(G1.getIdx());

            // Compute gate 0:  G0 = sigmoid(G0 + b0)
            uni_vmovss(G0s, G_addr(0));
            uni_vaddss(G0s, G0s, B_addr(0));
            sigmoid_injector_->compute_vector(G0s.getIdx());
            // we store it for use in postgemm_part2
            uni_vmovss(G_addr(0), G0s);

            // Compute gate 1: G1 = sigmoid(G1 + b1)
            uni_vmovss(G1s, G_addr(1));
            uni_vaddss(G1s, G1s, B_addr(1));
            sigmoid_injector_->compute_vector(G1s.getIdx());
            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(G_addr(1), G1);

            // states_t_l = states_tm1_l * G1
            uni_vmulss(G1s, G1s, ptr[addr_states_tm1_l_reg]);
            uni_vmovss(ptr[addr_states_t_l_reg], G1s);

            // increment address pointers
            add(addr_ws_gates_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, hstate_dt_size);

            // increment loop counter
            sub(loop_cnt, gate_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);

        }
        L(rem_loop_end_label);

        postamble();

        // Again, only one table is needed and shared between sigmoid and tanh
        sigmoid_injector_->prepare_table(true);
    }

};

template struct jit_uni_gru_cell_postgemm_part1_fwd<sse42, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd<avx2, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd<avx512_core, data_type::f32>;

}
}
}

#endif
