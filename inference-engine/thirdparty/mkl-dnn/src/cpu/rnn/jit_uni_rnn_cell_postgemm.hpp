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

#ifndef CPU_JIT_RNN_CELL_POSTGEMM
#define CPU_JIT_RNN_CELL_POSTGEMM

#include "jit_uni_rnn_common_postgemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t>
struct jit_uni_rnn_cell_postgemm_fwd: public jit_uni_rnn_postgemm
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_rnn_cell_postgemm_fwd)

    typedef typename utils::conditional<src_data_t == data_type::u8, int32_t,
            float>::type acc_data_t;
    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_rnn_cell_postgemm_fwd(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
    : jit_uni_rnn_postgemm(rnn, pd){}

    ~jit_uni_rnn_cell_postgemm_fwd(){
        delete injector_;
    }

    void init() override {
        // we use rax for constant tables
        injector_ = new injector_t(this, pd_->activation_kind(),
	        0.0f, 0.0f, true, rax);
        generate();
        kernel_ = (kernel_t) this->getCode();
    }

protected:
    injector_t *injector_;

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

        // Labels declaration
        Label vector_loop_start_label, vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 loop_cnt(r11);  // loop counter
        Reg64 table_reg(rbx); // table is used for data scale and shifts
        Reg64 tmp_reg(r12);   // used as temporary to customize mxcsr
        Reg64 weights_scales_reg(r13);
	// Here we do no unrolling, loop overhead should not be that dramatic
	// We skip vmm0 as it can be used by the injector for masks on sse4.2
	Vmm G(1), tmp1_vmm(5), tmp2_vmm(6), zero_vmm(7);

        // constant table map
        Address dscale_off_addr = ptr[table_reg];
        Address dshift_off_addr = ptr[table_reg + vlen];
        Address ymm_perm_mask_addr = ptr[table_reg + 2*vlen];
        Address zmm_perm_mask_addr = ptr[table_reg + 2*vlen + cpu_isa_traits<avx>::vlen];

        // quantize from float to u8
        auto q_d = [&](Vmm f, Vmm tmp_vmm, Reg64 tmp_reg) {
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

        };

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

        // dequantize from s32 to float
        auto deq_w = [&](Vmm s, Vmm tmp1, Vmm tmp2, int gate, bool packed) {
            // TODO: if mask is 0 precompute mul and inverse
            if (mask == 0)
                uni_vbroadcastss(tmp1, ptr[weights_scales_reg]);
            else
                uni_vmovups(tmp1, ptr[weights_scales_reg + gate * rnn_.dic * qscale_dt_size]);
            uni_vcvtdq2ps(s, s);
            uni_vmulps(tmp1, tmp1, dscale_off_addr);
            fast_recip(tmp1, tmp2, packed);
            uni_vmulps(s, s, tmp1);
        };

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_bias_reg = abi_param2;
        auto addr_states_t_l_reg = abi_param3;

        auto G_addr = ptr[addr_ws_gates_reg + 0 * rnn_.dic * gate_dt_size];
        auto B_addr = ptr[addr_bias_reg + 0 * rnn_.dic * bias_dt_size];

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        mov(weights_scales_reg, size_t(weights_scales));
	injector_->load_table_addr();

        mov(loop_cnt, rnn_.dic * gate_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            // load G
            uni_vmovups(G, G_addr);

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8){
                deq_w(G, tmp1_vmm, tmp2_vmm, 0, true);
            }

            // add biases
            uni_vmovups(tmp1_vmm, B_addr);
            uni_vaddps(G, G, tmp1_vmm);

            // inject eltwise code
	    injector_->compute_vector(G.getIdx());

        // if training we write back the gates
        if (pd_->desc()->prop_kind == prop_kind::forward_training)
            uni_vmovups(G_addr, G);

        // if int8, we quantize the resulting state
        if (src_data_t == data_type::u8) {
	        q_d(G, tmp1_vmm, tmp_reg);
            }

            // write back the result
            if(vlen_dst == vlen)
                uni_vmovups(ptr[addr_states_t_l_reg], G);
            else
                // we write only 1/4 of the register
                switch(vlen_dst){
                case 16: uni_vmovups(ptr[addr_states_t_l_reg], Xmm(G.getIdx())); break;
                case 8: uni_vmovsd(ptr[addr_states_t_l_reg], Xmm(G.getIdx())); break;
                case 4: uni_vmovss(ptr[addr_states_t_l_reg], Xmm(G.getIdx())); break;
                default:
                    assert(!"Unsuported vector length for quantization");
                }

            // increment address pointers
            add(addr_ws_gates_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
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
        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            // remaping registers to Xmms
            Xmm Gs(G.getIdx());
            Xmm tmp1s_vmm(tmp1_vmm.getIdx());

            // load G
            uni_vmovss(Gs, G_addr);

            // dequantize the gates from s32 to f32 if needed
            if (src_data_t == data_type::u8) {
                deq_w(G, tmp1_vmm, tmp2_vmm, 0, false);
            }

            // add biases
            uni_vmovss(tmp1s_vmm, B_addr);
            uni_vaddps(Gs, Gs, tmp1s_vmm);

            // inject eltwise code
	    injector_->compute_vector(Gs.getIdx());

            // if training we write back the gates
            if (pd_->desc()->prop_kind == prop_kind::forward_training)
                uni_vmovss(G_addr, Gs);

            // if int8, we quantize the resulting state
            if (src_data_t == data_type::u8) {
                q_d(G, tmp1_vmm, tmp_reg);
            }

	    switch(hstate_dt_size){
	    case 4: uni_vmovss(ptr[addr_states_t_l_reg], Gs); break;
            case 1: pextrb(ptr[addr_states_t_l_reg], Gs, 0x0); break;
	    default:
                assert(!"Unsuported vector length for quantization");
            }

            // increment address pointers
            add(addr_ws_gates_reg, gate_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            if (mask != 0)
                add(weights_scales_reg, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, gate_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);

        }
        L(rem_loop_end_label);

        postamble();

        // inject the constant table for the activation
        injector_->prepare_table();

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

template struct jit_uni_rnn_cell_postgemm_fwd<sse42, data_type::f32>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx512_core, data_type::f32>;

template struct jit_uni_rnn_cell_postgemm_fwd<sse42, data_type::u8>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx2, data_type::u8>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx512_core, data_type::u8>;

}
}
}

#endif
