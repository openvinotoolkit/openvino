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

#ifndef CPU_JIT_LSTM_CELL_POSTGEMM
#define CPU_JIT_LSTM_CELL_POSTGEMM

#include "jit_uni_rnn_common_postgemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t>
struct jit_uni_lstm_cell_postgemm_fwd: public jit_uni_rnn_postgemm
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd)

    typedef typename utils::conditional<src_data_t == data_type::u8, int32_t,
            float>::type acc_data_t;
    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_lstm_cell_postgemm_fwd(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
    : jit_uni_rnn_postgemm(rnn, pd){}

    ~jit_uni_lstm_cell_postgemm_fwd(){
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
    size_t cstate_dt_size = sizeof(float);
    size_t hstate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint8_t) : sizeof(float);
    size_t gate_dt_size = (src_data_t == data_type::u8) ? sizeof(uint32_t) : sizeof(float);
    size_t qscale_dt_size = sizeof(float);
    size_t bias_dt_size = sizeof(float);

    void generate() {
        using namespace Xbyak;

        const primitive_attr_t *attr = pd_->attr();
        int mask = attr->rnn_weights_qparams_.mask_;
        float *weights_scales = attr->rnn_weights_qparams_.scales_;
        float data_scale = attr->rnn_data_qparams_.scale_;
        float data_shift = attr->rnn_data_qparams_.shift_;
        round_mode_t rmode = attr->round_mode_;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 loop_cnt(r11);  // loop counter
        Reg64 table_reg(rbx); // table is used for data scale and shifts
        Reg64 tmp_reg(r12);   // used as temporary to customize mxcsr
        Reg64 weights_scales_reg(r13);
        // We skip vmm0 as it can be used by the injector for masks on sse4.2
        Vmm G0(1), G1(2), G2(3), G3(4), tmp1_vmm(5), tmp2_vmm(6), zero_vmm(7);

        // stack map
        Address saved_csr_addr = ptr[rsp];
        Address modified_csr_addr = ptr[rsp + sizeof(int64_t)];
        size_t stack_size = 2 * sizeof(int64_t);

        // constant table map
        Address dscale_off_addr = ptr[table_reg];
        Address dshift_off_addr = ptr[table_reg + vlen];
        Address ymm_perm_mask_addr = ptr[table_reg + 2*vlen];
        Address zmm_perm_mask_addr = ptr[table_reg + 2*vlen + cpu_isa_traits<avx>::vlen];

        // quantize from float to u8
        auto q_d = [&](Vmm f, Vmm tmp_vmm, Reg64 tmp_reg) {
            sub(rsp, stack_size);
            stmxcsr(saved_csr_addr); // save the mxcsr

            // set the rounding mode appropriatly
            mov(tmp_reg, saved_csr_addr);
            and_(tmp_reg, 0xffff9fff); // clear rc bits (rc = RNE)
            if (rmode == round_mode::down)
                or_(tmp_reg, 0x00002000); // set rc=01 if RD
            mov(modified_csr_addr, tmp_reg);
            ldmxcsr(modified_csr_addr);

            uni_vpxor(tmp_vmm, tmp_vmm, tmp_vmm);
            uni_vmulps(f, f, dscale_off_addr); // apply scale
            uni_vaddps(f, f, dshift_off_addr); // apply shift
            uni_vcvtps2dq(f, f); // convert to int32 with mxcsr rounding
            uni_vpackssdw(f, f, tmp_vmm); // convert from s32 to s16
            uni_vpackuswb(f, f, tmp_vmm); // convert from s16 to u8 with saturation
            // Note that the results are interleaved by 128 bit chunks, so we need to merge them together
            switch (vlen) {
            case 64:  { // Intel AVX-512
                Zmm fz(f.getIdx()), tmpz(tmp_vmm.getIdx());
                uni_vmovups(tmpz, zmm_perm_mask_addr);
                vpermd(fz, tmpz, fz);
                break; }
            case 32: { // Intel AVX
                Ymm fy(f.getIdx()), tmpy(tmp_vmm.getIdx());
                uni_vmovups(tmpy, ymm_perm_mask_addr);
                vpermd(fy, tmpy, fy);
                break; }
            case 16: // sse: nothing to do
                break;
            default: assert(!"Unsupported case");
            };

            ldmxcsr(saved_csr_addr); // restore the original mxcsr
            add(rsp, stack_size);
        };

// MKLDNN_ENABLE_FAST_RCP is not enabled by default
#ifdef MKLDNN_ENABLE_FAST_RCP
        auto fast_recip =[&](Vmm s, Vmm tmp, bool packed) {
            if (packed)
                uni_vrcpps(tmp, s);
            else
                uni_vrcpss(tmp, s); // prevent divide by zero
            // we add one Newton iteration
            uni_vmulps(s, s, tmp);
            uni_vmulps(s, s, tmp); // s <- s * tmp^2
            uni_vaddps(tmp, tmp, tmp);
            uni_vsubps(tmp, tmp, s);
            uni_vmovups(s, tmp); // s <- 2 * tmp - s * tmp^2
        };
#endif

        // dequantize from s32 to float
        auto deq_w = [&](Vmm s, Vmm tmp1, Vmm tmp2, int gate, bool packed) {
            // TODO: if mask is 0 precompute mul and inverse
            if (mask == 0)
                uni_vbroadcastss(tmp1, ptr[weights_scales_reg]);
            else {
                auto scales_ptr = ptr[weights_scales_reg
                    + gate * rnn_.dic * qscale_dt_size];
                if (packed)
                    uni_vmovups(tmp1, scales_ptr);
                else
                    uni_vmovss(tmp1, scales_ptr);
            }
            uni_vcvtdq2ps(s, s);
            uni_vmulps(tmp1, tmp1, dscale_off_addr);
#ifdef MKLDNN_ENABLE_FAST_RCP
            fast_recip(tmp1, tmp2, packed);
            uni_vmulps(s, s, tmp1);
#else
            uni_vdivps(s, s, tmp1);
#endif
        };

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;
        auto addr_c_states_tm1_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_c_states_t_l_reg = r10;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        mov(addr_c_states_t_l_reg, ptr[rsp + get_size_of_abi_save_regs() + 40]);
#else
        auto addr_c_states_t_l_reg = abi_param5;
#endif
        // helper lambda to address the gates and biases
        auto G_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dic * gate_dt_size];
        };
        auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dic * bias_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        mov(weights_scales_reg, size_t(weights_scales));
        // both sigmoid and tanh use the same table so load address just once in rax
        sigmoid_injector_->load_table_addr();

        mov(loop_cnt, rnn_.dic * gate_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // load G0 G1 G2 G3
            uni_vmovups(G0, G_addr(0));
            uni_vmovups(G1, G_addr(1));
            uni_vmovups(G2, G_addr(2));
            uni_vmovups(G3, G_addr(3));

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8){
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, true);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, true);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, true);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, true);
            }

            // add biases
            uni_vmovups(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(2));
            uni_vaddps(G2, G2, tmp1_vmm);
            uni_vmovups(tmp1_vmm, B_addr(3));
            uni_vaddps(G3, G3, tmp1_vmm);

            // inject eltwise code
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->compute_vector(G2.getIdx());
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training) {
                uni_vmovups(G_addr(0), G0);
                uni_vmovups(G_addr(1), G1);
                uni_vmovups(G_addr(2), G2);
                uni_vmovups(G_addr(3), G3);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmovups(tmp1_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1_vmm, tmp1_vmm, G1);
            uni_vfmadd231ps(tmp1_vmm, G0, G2);
            uni_vmovups(ptr[addr_c_states_t_l_reg], tmp1_vmm);

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->compute_vector(tmp1_vmm.getIdx());
            uni_vmulps(tmp1_vmm, tmp1_vmm, G3);

            // if int8, we quantize the resulting state
            if (src_data_t == data_type::u8) {
                q_d(tmp1_vmm, tmp2_vmm, tmp_reg);
            }

            // write back the result
            if(vlen_dst == vlen)
                uni_vmovups(ptr[addr_states_t_l_reg], tmp1_vmm);
            else
                // we write only 1/4 of the register
                switch(vlen_dst){
                case 16: uni_vmovups(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                case 8: uni_vmovsd(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                case 4: uni_vmovss(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx())); break;
                default: assert(!"Unsuported vector length for quantization");
                }

            // increment address pointers
            add(addr_ws_gates_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_c_states_tm1_l_reg, vlen);
            add(addr_c_states_t_l_reg, vlen);
            if (mask != 0)
                add(weights_scales_reg, vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        L(rem_loop_start_label);
        {
            // load G0 G1 G2 G3
            uni_vmovss(G0, G_addr(0));
            uni_vmovss(G1, G_addr(1));
            uni_vmovss(G2, G_addr(2));
            uni_vmovss(G3, G_addr(3));

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8){
                deq_w(G0, tmp1_vmm, tmp2_vmm, 0, false);
                deq_w(G1, tmp1_vmm, tmp2_vmm, 1, false);
                deq_w(G2, tmp1_vmm, tmp2_vmm, 2, false);
                deq_w(G3, tmp1_vmm, tmp2_vmm, 3, false);
            }

            // add biases
            uni_vmovss(tmp1_vmm, B_addr(0));
            uni_vaddps(G0, G0, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(1));
            uni_vaddps(G1, G1, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(2));
            uni_vaddps(G2, G2, tmp1_vmm);
            uni_vmovss(tmp1_vmm, B_addr(3));
            uni_vaddps(G3, G3, tmp1_vmm);

            // inject eltwise code
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->compute_vector(G2.getIdx());
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training) {
                uni_vmovss(G_addr(0), G0);
                uni_vmovss(G_addr(1), G1);
                uni_vmovss(G_addr(2), G2);
                uni_vmovss(G_addr(3), G3);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmovups(tmp1_vmm, ptr[addr_c_states_tm1_l_reg]);
            uni_vmulps(tmp1_vmm, tmp1_vmm, G1);
            uni_vfmadd231ps(tmp1_vmm, G0, G2);
            uni_vmovss(ptr[addr_c_states_t_l_reg], tmp1_vmm);

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->compute_vector(tmp1_vmm.getIdx());
            uni_vmulps(tmp1_vmm, tmp1_vmm, G3);

            // if int8, we quantize the resulting state
            if (src_data_t == data_type::u8) {
                q_d(tmp1_vmm, tmp2_vmm, tmp_reg);
            }

            // write back the result
            switch(hstate_dt_size) {
            case 4: uni_vmovss(ptr[addr_states_t_l_reg], tmp1_vmm); break;
            case 1: pextrb(ptr[addr_states_t_l_reg], Xmm(tmp1_vmm.getIdx()), 0x0); break;
            default: assert(!"Unsuported vector length for quantization");
            }

            // increment address pointers
            add(addr_ws_gates_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_c_states_tm1_l_reg, cstate_dt_size);
            add(addr_c_states_t_l_reg, cstate_dt_size);
            if (mask != 0)
                add(weights_scales_reg, qscale_dt_size);

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
            for (size_t i = 0; i < vlen / sizeof(float); i++) dd(float2int(data_scale));
            for (size_t i = 0; i < vlen / sizeof(float); i++) dd(float2int(data_shift));
            // perm mask for ymm
            dd(0); dd(4); dd(2); dd(3); dd(1); dd(5); dd(6); dd(7);
            // perm mask for zmm
            dd(0); dd(4); dd(8); dd(12); dd(1); dd(5); dd(6); dd(7);
            dd(2); dd(9); dd(10); dd(11); dd(3); dd(12); dd(13); dd(14);
        }
    }

};

template struct jit_uni_lstm_cell_postgemm_fwd<sse42, data_type::f32>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx512_core, data_type::f32>;

template struct jit_uni_lstm_cell_postgemm_fwd<sse42, data_type::u8>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx2, data_type::u8>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx512_core, data_type::u8>;

}
}
}

#endif
