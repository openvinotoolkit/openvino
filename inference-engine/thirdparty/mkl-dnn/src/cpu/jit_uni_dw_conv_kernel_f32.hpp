/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef JIT_UNI_DW_CONV_KERNEL_F32_HPP
#define JIT_UNI_DW_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_dw_conv_fwd_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_fwd_kernel_f32)

    jit_uni_dw_conv_fwd_kernel_f32(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr): jcp(ajcp), attr_(attr) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_dw_conv_fwd_kernel_f32() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            bool with_relu = false, float relu_negative_slope = 0.f);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = (isa == sse42)
        ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t aux1_reg_kernel = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_ch_blocks = aux1_reg_input;
    reg64_t imm_addr64 = aux1_reg_input;

    reg64_t reg_d_weights = imm_addr64;
    reg64_t reg_d_bias = iter_kh;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    inline void load_src(int ur_ch_blocks, int ur_w);
    inline void apply_filter(int ur_ch_blocks, int ur_w);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w);
    inline void apply_postprocess(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w);
    inline void loop_body(int ur_ch_blocks);

    void generate();

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_data_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_data_kernel_f32)

    jit_uni_dw_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_ddst   = r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t aux1_reg_kernel = rbp;
    reg64_t reg_dsrc       = rsi;

    reg64_t reg_ur_str_w = r9;
    reg64_t reg_ch_blocks = rbx;

    reg64_t iter_kh = r11;
    reg64_t iter_kw = r12;
    reg64_t reg_kh  = r13;
    reg64_t reg_kw  = r14;

    inline void loop_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w);

    void generate();
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_weights_kernel_f32 : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_weights_kernel_f32)

    jit_uni_dw_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_dw_conv_call_s *)) this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_dw_conv_call_s *);

private:
    //using Vmm = Xbyak::Zmm;
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using te_size
            = unsigned char; /* set the 'table_entry' data size. For this
                                implementation, only values > 255 are needed. */
    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int reg_repeats = (isa == sse42) ? 2 : 1;
    inline void write_table(te_size data) { db(data); }
    //const Xbyak::AddressFrame &vmmword = zword;
    const Xbyak::AddressFrame &vmmword
            = (isa == sse42) ? xword : (isa == avx2) ? yword : zword;

    /* XXX: offset between input and accummulators is 3, therefore, assume 'kw'
     * is no larger than 3*/
    inline Vmm get_bias_reg(int idx = 0) { return Vmm(idx); }
    inline Vmm get_output_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_input_reg(int idx) { return Vmm(idx + 4 * reg_repeats + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 1 * reg_repeats + 1); }
    inline Vmm get_aux_reg() { return Vmm(0); }

    reg64_t tmp_reg_idx_input = r8;
    reg64_t tmp_reg_kh_input = r9;
    reg64_t tmp_reg_idx_output = r10;
    reg64_t tmp_reg_filter = r11;

    /* parameter passed by driver into kernel */
    reg64_t reg_table_flags = rbx;
    Xbyak::Reg8 reg_table_idx = bl;
    Xbyak::Reg8 reg_exec_flag = bh;

    /* holds the address for the 'bounds table' that is generated during JIT */
    reg64_t reg_bound_table_addr = r13;

    reg64_t reg_tmp_off = rax;
    Xbyak::Reg8 reg_tmp_al = al;

    reg64_t iter_oh = rdx;
    Xbyak::Reg8 iter_oh_lb = dl;
    reg64_t kh_offset = rdx;
    Xbyak::Reg8 kh_offset_lb = dl;

    reg64_t iter_ow_blk = rbp;
    reg64_t iter_kh  = rsi;

    /* Base addresses for convolution parameters. */
    reg64_t reg_input_baddr = r15;
    reg64_t reg_output_baddr = r12;
    reg64_t reg_filter_baddr = abi_not_param1;
    reg64_t reg_bias_baddr = r14;

    Xbyak::Label bound_start_table;

    /* Return the amount of blocks to execute depending on the convolution
     * dimensions and block_size e.g.
     *  {ow = 112, ow_block_size = 14} -> requires:
     *      1 left block,
     *      1 middle block,
     *      1 right block;
     * {ow = 28, ow_block_size = * 14} -> requires:
     *      1 left block,
     *      1 right block. */
    inline int get_loop_bounds_count(
            const int padding, const int h_dimension, const int block_size) {
        const int num_top_padded_blk = utils::div_up(padding, block_size);
        const int num_tail_blk
                = (h_dimension - num_top_padded_blk * block_size > 0) ? 1 : 0;
        const int num_middle_blk
                = (h_dimension
                    - (num_top_padded_blk + num_tail_blk) * block_size
                          > 0) ? 1 : 0;
        return num_top_padded_blk + num_middle_blk + num_tail_blk;
    }

    /* Create a table containing the values that define the kernel's loop
     * behavior. The purpose of using this table is to eliminate the
     * implementation complexities and performance impact of in-execution
     * computation of loop bounds in regards to stride and padding.  The table
     * consists of 3 sections:
     * 1) Initial Bounds for 'oh' loop.
     * 2) Input address offset flag: '1' indicates an input address increment,
     *    '0' results in no increment.
     * 3) End-bounds for 'oh' loop.
     *
     * The table is written into memory as the following format:
     * Filter_size:    |--- kh ---|
     * Table:           __________
     * 1st section:    |          |
     *                 |- - - - - |
     * 2nd section:    |          |
     *                 |- - - - - |
     * 3rd section:    |__________|
     *
     * Example for convolution: ih=112, oh=112, kh=3, ph=1
     *   __________
     *  | 1,  0,  0| -> upper 'oh' loop initial bounds
     *  | 0,  0,  0| -> middle 'oh' loop initial bounds
     *  | 0,  0,  0| -> bottom loop initial bounds
     *  |----------|
     *  | 0,  1,  0| -> *There is no input offset for kh = 0, i.e. the
     *  | 1,  1,  1|    offset_flag is '0' becase of padding.
     *  | 1,  1,  1|
     *  |----------|
     *  |14, 14, 14| -> lower 'oh' loop end bounds
     *  |14, 14, 14| -> (etc)
     *  |14, 14, 13| -> *The last 'kh' loop has an upper bound of 13
     *  |__________|    because of padding.
     *    0,  1,  2  -> kh values
     * */
    inline void create_h_bounds_table();

    /* Micro-kernel JIT'ing, fusing 'kw' and 'ow_block' loops into unrolled FMAs
     */
    inline void compute_ow_step_unroll(
            int l_pad, int r_pad, int pad_offset, int ow_block);

    /* JIT'ing the outer loops for the micro-kernel -> {kh, oh_block} */
    inline void compute_kh_loop(int l_pad, int r_pad, int pad_offset,
            bool first_iteration, int ow_block = 0);

    /* Write 'width' micro-kernel JITs; depending on the padding and convolution
     * size, write a micro-kernel for the left ow-block, middle ow-block(s), and
     * right ow-block.*/
    inline void compute_ow_block_unroll();

    inline void load_filter();
    inline void zero_filter();
    inline void load_bias();
    inline void zero_bias();
    inline void compute_bias_step_unroll(const int unroll_w);
    inline void compute_bias_loop();
    inline void store_filter();
    inline void store_bias();

    void generate();
};
}
}
}

#endif
