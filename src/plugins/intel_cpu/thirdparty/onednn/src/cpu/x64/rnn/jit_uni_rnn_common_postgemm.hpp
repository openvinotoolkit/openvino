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

#ifndef CPU_X64_RNN_JIT_UNI_RNN_COMMON_POSTGEMM_HPP
#define CPU_X64_RNN_JIT_UNI_RNN_COMMON_POSTGEMM_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/rnn_pd.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_rnn_postgemm : public jit_generator {

    jit_uni_rnn_postgemm(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : rnn_(rnn)
        , pd_(pd)
        , projection_(false)
        , dscale_off_addr(0)
        , dshift_off_addr(0)
        , ymm_perm_mask_addr(0)
        , zmm_perm_mask_addr(0)
        , zero_addr(0)
        , u8_saturation_addr(0)
        , weights_scales_reg(r13)
        , qtable(r14)
        , qd_reg_idx(15)
        , bf16_reg1(zmm31)
        , bf16_reg2(zmm30)
        , bf16_reg3(zmm29)
        , bf16_reg4(r13)
        , bf16_reg5(zmm28)
        , bf16_k_mask(k2)
        , bf16_dq_reg_idx(15)
        , bias_dt_size_(types::data_type_size(rnn.bias_dt))
        , cstate_dt_size_(types::data_type_size(rnn.src_iter_c_dt)) {}

    ~jit_uni_rnn_postgemm() {
        if (bf16_emu_) delete bf16_emu_;
    }

    bool is_projection() const { return projection_; };

    virtual status_t init(data_type_t src_data_t) {
        // no need to check as bf16 is guarded for avx512 and above in rnn primtive
        using namespace Xbyak;
        if (src_data_t == data_type::bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(this, bf16_reg1, bf16_reg2,
                    bf16_reg3, bf16_reg4, bf16_reg5);

        } else
            bf16_emu_ = nullptr;
        return status::success;
    }

    template <typename dst_layer_t, typename dst_iter_t, typename src_iter_t,
            typename gemm_acc_t, typename gates_t, typename scratch_t>
    rnn_postgemm_sig(execute) {
        if (pd_->desc()->prop_kind == prop_kind::backward)
            execute_bwd(rnn, cell_position, ws_gates_, scratch_gates_,
                    dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
                    diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                    diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                    weights_peephole_, bias_, ws_grid_, scratch_cell_,
                    dst_iter_, weights_scales_, block_step);
        else
            execute_fwd(rnn, cell_position, ws_gates_, scratch_gates_,
                    dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
                    diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                    diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                    weights_peephole_, bias_, ws_grid_, scratch_cell_,
                    dst_iter_, weights_scales_, block_step);
    }

    template <typename dst_layer_t, typename dst_iter_t, typename src_iter_t,
            typename gemm_acc_t, typename gates_t, typename scratch_t>
    rnn_postgemm_sig(execute_fwd) {
        using namespace rnn_utils;
        if (rnn.is_brgemm && !rnn_.unfused_post_gemm) {
            for (int i = 0; i < rnn.m_block; i++)
                postgemm_fwd_call(i, rnn, cell_position, ws_gates_,
                        scratch_gates_, dst_layer_, dst_iter_c_, src_iter_,
                        src_iter_c_, weights_peephole_, bias_, ws_grid_,
                        scratch_cell_, dst_iter_, weights_scales_, block_step);
        } else {
            // Todo: add parallelization on dhc for the batch 1 case
            // Assumption: the kernel runs a loop on dhc elements
            parallel_nd(rnn.mb, [&](dim_t i) {
                postgemm_fwd_call(i, rnn, cell_position, ws_gates_,
                        scratch_gates_, dst_layer_, dst_iter_c_, src_iter_,
                        src_iter_c_, weights_peephole_, bias_, ws_grid_,
                        scratch_cell_, dst_iter_, weights_scales_, 0);
            });
        }
    }

    template <typename dst_layer_t, typename dst_iter_t, typename src_iter_t,
            typename gates_t, typename scratch_t>
    inline void postgemm_fwd_call(int m, const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, gates_t *ws_gates_,
            scratch_t *scratch_gates_, dst_layer_t *dst_layer_,
            void *dst_iter_c_, const src_iter_t *src_iter_,
            const void *src_iter_c_, const float *weights_peephole_,
            const void *bias_, gates_t *ws_grid_, scratch_t *scratch_cell_,
            dst_iter_t *dst_iter_, float *weights_scales_,
            int block_step) const {
        const rnn_utils::ws_gates_aoc<gates_t> ws_gates(rnn, ws_gates_);
        const rnn_utils::scratch_gates_aoc<scratch_t> scratch_gates(
                rnn, scratch_gates_);
        const rnn_utils::weights_peephole_aoc_t<const float> weights_peephole(
                rnn, weights_peephole_);
        const auto bias = rnn_utils::make_raw_aoc(
                bias_, types::data_type_size(rnn.bias_dt), rnn.n_bias, rnn.dhc);

        const auto src_iter_ld = rnn.src_iter_ld(cell_position);
        const int dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
        const auto dst_layer_ld
                = rnn.dst_layer_ld(cell_position, is_projection());
        const auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
        const int src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

        const rnn_utils::ws_states_layer_aoc<dst_layer_t> dst_layer(
                rnn, dst_layer_, dst_layer_ld);
        const rnn_utils::ws_states_iter_aoc<dst_iter_t> dst_iter(
                rnn, dst_iter_, dst_iter_ld);
        const rnn_utils::ws_states_iter_aoc<const src_iter_t> src_iter(
                rnn, src_iter_, src_iter_ld);
        const auto dst_iter_c = rnn_utils::make_raw_aoc(dst_iter_c_,
                types::data_type_size(rnn.dst_iter_c_dt),
                rnn.ws_states_iter_c_nld, dst_iter_c_ld);
        const auto src_iter_c = rnn_utils::make_raw_aoc(src_iter_c_,
                types::data_type_size(rnn.src_iter_c_dt),
                rnn.ws_states_iter_c_nld, src_iter_c_ld);
        const rnn_utils::ws_gates_aoc<scratch_t> scratch_cell(
                rnn, scratch_cell_);
        const utils::array_offset_calculator<gates_t, 2> ws_Wh_b(
                ws_grid_, rnn.mb, rnn.dhc);

// Since the function F(...) returns by reference so an exception has
// to be made for nullptr argument
#define SAFE_PTR(F, ...) (CONCAT2(F, _) ? &(F(__VA_ARGS__)) : nullptr)

        void *param1_ = SAFE_PTR(ws_gates, m, 0, 0); // RNN, LSTM, GRU
        void *param2_ = SAFE_PTR(scratch_gates, m, 0, 0); // RNN, LSTM, GRU
        const void *param3_ = bias(0, 0); // RNN, LSTM, GRU
        void *param4_ = SAFE_PTR(dst_layer, m, 0); // RNN, LSTM, GRU
        void *param5_ = SAFE_PTR(dst_iter, m, 0); // RNN, LSTM, GRU
        const void *param6_;
        void *param7_, *param8_;
        void *param9_ = (void *)weights_scales_;
        const size_t param10_ = block_step;

        switch (pd_->cell_kind()) {
            case alg_kind::vanilla_lstm:
                param6_ = is_projection() ? src_iter_c_ : src_iter_c(m, 0);
                param7_ = const_cast<void *>(dst_iter_c(m, 0));
                param8_ = (void *)SAFE_PTR(weights_peephole, 0, 0);
                break;
            case alg_kind::lbr_gru:
                param6_ = SAFE_PTR(src_iter, m, 0);
                param7_ = SAFE_PTR(scratch_cell, m, 0, 0);
                param8_ = ws_grid_ ? &ws_Wh_b(m, 0) : nullptr;
                break;
            case alg_kind::vanilla_gru:
                param6_ = SAFE_PTR(src_iter, m, 0);
                param7_ = nullptr;
                param8_ = nullptr;
                break;
            default:
                param6_ = nullptr;
                param7_ = nullptr;
                param8_ = nullptr;
                break;
        }
        this->operator()(param1_, param2_, param3_, param4_, param5_, param6_,
                param7_, param8_, param9_, param10_);
#undef SAFE_PTR
    }

    template <typename dst_layer_t, typename dst_iter_t, typename src_iter_t,
            typename gemm_acc_t, typename gates_t, typename scratch_t>
    rnn_postgemm_sig(execute_bwd) {
        using namespace rnn_utils;
        const int dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
        const int src_iter_c_ld = rnn.src_iter_c_ld(cell_position);
        const auto src_iter_ld = rnn.src_iter_ld(cell_position);

        const rnn_utils::weights_peephole_aoc_t<const float> weights_peephole(
                rnn, weights_peephole_);
        const rnn_utils::ws_gates_aoc<gates_t> ws_gates(rnn, ws_gates_);
        const rnn_utils::ws_gates_aoc<scratch_t> scratch_gates(
                rnn, scratch_gates_);
        const rnn_utils::ws_diff_states_layer_aoc<gemm_acc_t> diff_src_layer(
                rnn, diff_src_layer_);
        const rnn_utils::ws_diff_states_iter_aoc<gemm_acc_t> diff_src_iter(
                rnn, diff_src_iter_);
        const rnn_utils::ws_diff_states_iter_c_aoc<gemm_acc_t> diff_src_iter_c(
                rnn, diff_src_iter_c_);
        const rnn_utils::ws_diff_states_layer_aoc<gemm_acc_t> diff_dst_layer(
                rnn, diff_dst_layer_);
        const rnn_utils::ws_diff_states_iter_aoc<gemm_acc_t> diff_dst_iter(
                rnn, diff_dst_iter_);
        const rnn_utils::ws_diff_states_iter_c_aoc<gemm_acc_t> diff_dst_iter_c(
                rnn, diff_dst_iter_c_);
        const auto dst_iter_c = rnn_utils::make_raw_aoc(dst_iter_c_,
                types::data_type_size(rnn.dst_iter_c_dt),
                rnn.ws_states_iter_c_nld, dst_iter_c_ld);
        const auto src_iter_c = rnn_utils::make_raw_aoc(src_iter_c_,
                types::data_type_size(rnn.src_iter_c_dt),
                rnn.ws_states_iter_c_nld, src_iter_c_ld);
        const ws_states_iter_aoc<const src_iter_t> src_iter(
                rnn, src_iter_, src_iter_ld);
        const ws_gates_aoc<scratch_t> scratch_cell(rnn, scratch_cell_);
        const utils::array_offset_calculator<scratch_t, 2> hG1(
                scratch_cell_, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld);
        const utils::array_offset_calculator<gates_t, 2> ws_grid(
                ws_grid_, rnn.mb, rnn.dhc);
// Since the function F(...) returns by reference so an exception has
// to be made for nullptr argument
#define SAFE_PTR(F, ...) (CONCAT2(F, _) ? &(F(__VA_ARGS__)) : nullptr)
        // Todo: add parallelization on dhc for the batch 1 case
        // Assumption: the kernel runs a loop on dhc elements
        parallel_nd(rnn.mb, [&](dim_t i) {
            void *param1_, *param2_, *param4_, *param5_, *param7_, *param8_,
                    *param9_;
            const void *param3_, *param6_;
            static constexpr size_t param10_ = 0;
            switch (pd_->cell_kind()) {
                case alg_kind::vanilla_lstm:
                    param1_ = SAFE_PTR(ws_gates, i, 0, 0);
                    param2_ = SAFE_PTR(scratch_gates, i, 0, 0); //RNN, LSTM, GRU
                    param3_ = SAFE_PTR(diff_dst_layer, i, 0);
                    param4_ = SAFE_PTR(diff_dst_iter, i, 0);
                    param5_ = SAFE_PTR(diff_src_iter_c, i, 0);
                    param6_ = SAFE_PTR(diff_dst_iter_c, i, 0);
                    param7_ = const_cast<void *>(src_iter_c(i, 0));
                    param8_ = const_cast<void *>(dst_iter_c(i, 0));
                    param9_ = (void *)SAFE_PTR(weights_peephole, 0, 0);
                    break;
                case alg_kind::lbr_gru:
                    param1_ = SAFE_PTR(ws_gates, i, 0, 0);
                    param2_ = SAFE_PTR(scratch_gates, i, 0, 0);
                    param3_ = SAFE_PTR(diff_dst_layer, i, 0);
                    param4_ = SAFE_PTR(diff_dst_iter, i, 0);
                    param5_ = SAFE_PTR(diff_src_iter, i, 0);
                    param6_ = SAFE_PTR(src_iter, i, 0);
                    param7_ = SAFE_PTR(scratch_cell, i, 0, 0);
                    param8_ = SAFE_PTR(ws_grid, i, 0);
                    param9_ = nullptr;
                    break;
                case alg_kind::vanilla_gru:
                    // TODO: split part 1 and part2 APIs/ABIs
                    param1_ = SAFE_PTR(ws_gates, i, 0, 0);
                    param2_ = SAFE_PTR(scratch_gates, i, 0, 0); //RNN, LSTM, GRU
                    param3_ = SAFE_PTR(diff_dst_layer, i, 0); // non part2
                    param4_ = SAFE_PTR(diff_dst_iter, i, 0); // non part2
                    param5_ = SAFE_PTR(diff_src_iter, i, 0);
                    param6_ = SAFE_PTR(src_iter, i, 0);
                    param7_ = scratch_cell_ ? &hG1(i, 0) : nullptr; // non part1
                    param8_ = SAFE_PTR(ws_grid, i, 0); // non part1
                    param9_ = SAFE_PTR(diff_src_layer, i, 0); // non part1
                    break;
                case alg_kind::vanilla_rnn:
                    param1_ = SAFE_PTR(ws_gates, i, 0, 0);
                    param2_ = SAFE_PTR(scratch_gates, i, 0, 0);
                    param3_ = SAFE_PTR(diff_dst_layer, i, 0);
                    param4_ = SAFE_PTR(diff_dst_iter, i, 0);
                    param5_ = nullptr;
                    param6_ = nullptr;
                    param7_ = nullptr;
                    param8_ = nullptr;
                    param9_ = nullptr;
                    break;
                default:
                    assert(!"unsupported");
                    param1_ = nullptr;
                    param2_ = nullptr;
                    param3_ = nullptr;
                    param4_ = nullptr;
                    param5_ = nullptr;
                    param6_ = nullptr;
                    param7_ = nullptr;
                    param8_ = nullptr;
                    param9_ = nullptr;
                    break;
            }
            this->operator()(param1_, param2_, param3_, param4_, param5_,
                    param6_, param7_, param8_, param9_, param10_);
        });
#undef SAFE_PTR
    }

protected:
    void init_regs(float *weights_scales, size_t vlen) {
        switch (pd_->weights_md()->data_type) {
            case data_type::bf16: {
                /* bfloat downconvert init */
                if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
                /* init mask for upconvert */
                mov(r13d, 1);
                kmovd(bf16_k_mask, r13d);
                break;
            }
            case data_type::s8: {
                /* int8 (de)quantization init*/
                mov(qtable, qlabel);
                if (rnn_.is_brgemm && !rnn_.unfused_post_gemm) {
                    auto base_args = get_stack_params_address();
                    // Read param #9
#ifdef _WIN32
                    mov(weights_scales_reg, ptr[base_args + 32]);
#else
                    mov(weights_scales_reg, ptr[base_args + 16]);
#endif
                } else {
                    float *weights_scales
                            = pd_->attr()->rnn_weights_qparams_.scales_;
                    mov(weights_scales_reg, size_t(weights_scales));
                }

                zero_addr = ptr[qtable];
                u8_saturation_addr = ptr[qtable + vlen];
                dscale_off_addr = ptr[qtable + 2 * vlen];
                dshift_off_addr = ptr[qtable + 3 * vlen];
                ymm_perm_mask_addr = ptr[qtable + 4 * vlen];
                zmm_perm_mask_addr
                        = ptr[qtable + 4 * vlen + cpu_isa_traits<avx>::vlen];
                break;
            }
            case data_type::f32: {
                break;
            }
            default: assert(!"not supported");
        }
    }

    void init_regs(size_t vlen) {
        assert(pd_->weights_md()->data_type != data_type::s8);
        return init_regs(nullptr, vlen);
    };

    void init_table(size_t vlen) {
        if (pd_->weights_md()->data_type != data_type::s8) return;
        /* int8 (de)quantization init*/
        const primitive_attr_t *attr = pd_->attr();
        const float data_scale = attr->rnn_data_qparams_.scale_;
        const float data_shift = attr->rnn_data_qparams_.shift_;

        L(qlabel);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(0.0f));
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(255.0f));
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(data_scale));
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(data_shift));
            // perm mask for ymm
            dd(0);
            dd(4);
            dd(2);
            dd(3);
            dd(1);
            dd(5);
            dd(6);
            dd(7);
            // perm mask for zmm
            dd(0);
            dd(4);
            dd(8);
            dd(12);
            dd(1);
            dd(5);
            dd(6);
            dd(7);
            dd(2);
            dd(9);
            dd(10);
            dd(11);
            dd(3);
            dd(12);
            dd(13);
            dd(14);
        }
    }

    void inc_regs(int mask, size_t vlen) {
        if (pd_->weights_md()->data_type == data_type::s8) {
            if (mask != 0) add(weights_scales_reg, vlen);
        }
    }
    void inc_regs(size_t vlen) {
        assert(pd_->weights_md()->data_type != data_type::s8);
        inc_regs(0, vlen);
    }

    template <typename Vmm>
    void fast_recip(Vmm s, Vmm tmp, bool packed) {
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
    }

    // quantize from float to u8
    // Assumption: write_only = true assumes that the quantized value
    // to write is in src
    template <typename Vmm>
    void q_d(data_type_t src_data_t, Xbyak::Address dst, Vmm src, int in_len,
            bool write_only = false) {
        Vmm qd_vmm(qd_reg_idx);
        if (!write_only) {
            uni_vpxor(qd_vmm, qd_vmm, qd_vmm);
            uni_vmulps(src, src, dscale_off_addr); // apply scale
            uni_vaddps(src, src, dshift_off_addr); // apply shift
            // To saturate properly, we use min/max on the float value
            uni_vmaxps(src, src, zero_addr);
            uni_vminps(src, src, u8_saturation_addr);
            uni_vcvtps2dq(src, src); // convert to int32
            uni_vpackssdw(src, src, qd_vmm); // convert from s32 to s16
            // convert from s16 to u8/s8 with saturation
            if (src_data_t == data_type::u8)
                uni_vpackuswb(src, src, qd_vmm);
            else
                uni_vpacksswb(src, src, qd_vmm);
        }
        // Note that the results are interleaved by 128 bit chunks, so we need to merge them together
        switch (in_len) {
            case 64: { // Intel AVX-512
                if (!write_only) {
                    Xbyak::Zmm srcz(src.getIdx()), tmpz(qd_vmm.getIdx());
                    uni_vmovups(tmpz, zmm_perm_mask_addr);
                    vpermd(srcz, tmpz, srcz);
                }
                uni_vmovups(dst, Xbyak::Xmm(src.getIdx()));
                break;
            }
            case 32: { // Intel AVX
                if (!write_only) {
                    Xbyak::Ymm srcy(src.getIdx()), tmpy(qd_vmm.getIdx());
                    uni_vmovups(tmpy, ymm_perm_mask_addr);
                    vpermd(srcy, tmpy, srcy);
                }
                uni_vmovsd(dst, Xbyak::Xmm(src.getIdx()));
                break;
            }
            case 16: // sse: nothing to do
                uni_vmovss(dst, Xbyak::Xmm(src.getIdx()));
                break;
            case 4: uni_vpextrb(dst, Xbyak::Xmm(src.getIdx()), 0x0); break;

            default: assert(!"unsupported case");
        };
    }

    // dequantize from s32 to float
    template <typename Vmm>
    void deq_w(data_type_t src_data_t, Vmm s, Vmm tmp1, Vmm tmp2,
            dim_t scale_off, int mask, bool packed,
            Xbyak::Reg64 *comp = nullptr) {
        // nothing to do if not int8
        if (!utils::one_of(src_data_t, data_type::u8, data_type::s8)) return;

        size_t qscale_dt_size = sizeof(float);

        // TODO: if mask is 0 precompute mul and inverse
        if (mask == 0)
            uni_vbroadcastss(tmp1, ptr[weights_scales_reg]);
        else {
            auto scales_ptr
                    = ptr[weights_scales_reg + scale_off * qscale_dt_size];
            if (packed)
                uni_vmovups(tmp1, scales_ptr);
            else
                uni_vmovss(tmp1, scales_ptr);
        }
        uni_vcvtdq2ps(s, s);
        // Here we subtract a compensation if need be
        if (comp) { uni_vsubps(s, s, ptr[*comp]); }
        uni_vmulps(tmp1, tmp1, dscale_off_addr);
#ifdef DNNL_ENABLE_FAST_RCP
        fast_recip(tmp1, tmp2, packed);
        uni_vmulps(s, s, tmp1);
#else
        uni_vdivps(s, s, tmp1);
#endif
    }

    // dequantize from u8 to float
    template <typename Vmm>
    void deq_h(Vmm dst, Xbyak::Address src, int in_len) {
        if (4 == in_len) {
            uni_vpinsrb(dst, dst, src, 0x0);
            uni_vpmovzxbd(dst, dst);
        } else {
            uni_vpmovzxbd(dst, src);
        }
        uni_vcvtdq2ps(dst, dst);
        uni_vsubps(dst, dst, dshift_off_addr);
        uni_vdivps(dst, dst, dscale_off_addr);
    }

    // upconvert from bf16 to float
    template <typename Vmm>
    void bf16_uc(Vmm dst, Xbyak::Address src, int in_len) {
        switch (in_len) {
            case 64:
                vpmovzxwd(dst, src);
                vpslld(dst, dst, 0x10);
                break;
            case 4:
                vpmovzxwd(dst | bf16_k_mask | T_z, src);
                vpslld(dst, dst, 0x10);
                break;
            default: assert(!"unsupported");
        }
    }

    // downconvert from float to bf16
    // Assumption: write_only = true assumes that we want to
    // immediately rewrite the downconverted result that is still in
    // bf16_dq_reg_idx
    template <typename Vmm>
    void bf16_dc(
            Xbyak::Address dst, Vmm src, int in_len, bool write_only = false) {
        Xbyak::Zmm srcz(src.getIdx());
        Xbyak::Ymm bf16_reg_dc(bf16_dq_reg_idx);
        if (!write_only) {
            if (bf16_emu_)
                bf16_emu_->vcvtneps2bf16(bf16_reg_dc, srcz);
            else
                vcvtneps2bf16(bf16_reg_dc, srcz);
        }
        switch (in_len) {
            case 64: uni_vmovups(dst, bf16_reg_dc); break;
            case 4:
                uni_vpextrw(dst, Xbyak::Xmm(bf16_reg_dc.getIdx()), 0x0);
                break;
            default: assert(!"unsupported case");
        }
    }

    // handles quantization/conversion and write to memory
    // Assumption: write_only = true assumes that
    // 1. to_src was already called with the same source and with
    // write_only = false.
    // 2. the src register and the temporary registers for
    // quantization/downconvert were not overritten in between the two
    // calls
    template <typename Vmm>
    void to_src(const Xbyak::Address &dst, const Vmm &src, data_type_t src_dt,
            int in_len, bool write_only = false) {
        switch (src_dt) {
            case data_type::f32:
                if (in_len == (int)src.getBit() / 8)
                    uni_vmovups(dst, src);
                else if (in_len == 4)
                    uni_vmovss(dst, src);
                else
                    assert(!"unsupported");
                break;
            case data_type::bf16: bf16_dc(dst, src, in_len, write_only); break;
            case data_type::u8:
            case data_type::s8:
                q_d(src_dt, dst, src, in_len, write_only);
                break;
            default: assert(!"unsupported");
        }
    }

    template <typename Vmm>
    void to_float(const Vmm &dst, const Xbyak::Address &src, data_type_t src_dt,
            int in_len) {
        switch (src_dt) {
            case data_type::f32:
                if (in_len == (int)dst.getBit() / 8)
                    uni_vmovups(dst, src);
                else if (in_len == 4)
                    uni_vmovss(dst, src);
                else
                    assert(!"unsupported");
                break;
            case data_type::bf16: bf16_uc(dst, src, in_len); break;
            case data_type::u8:
            case data_type::s8: deq_h(dst, src, in_len); break;
            default: assert(!"unsupported");
        }
    }

    const rnn_utils::rnn_conf_t &rnn_;
    const rnn_pd_t *pd_;
    bool projection_;
    bf16_emulation_t *bf16_emu_ = nullptr;

    // registers/Labels used for int8 quantization and conversions
    Xbyak::Address dscale_off_addr;
    Xbyak::Address dshift_off_addr;
    Xbyak::Address ymm_perm_mask_addr;
    Xbyak::Address zmm_perm_mask_addr;
    Xbyak::Address zero_addr;
    Xbyak::Address u8_saturation_addr;
    Xbyak::Reg64 weights_scales_reg;
    Xbyak::Reg64 qtable;
    Xbyak::Label qlabel;
    int qd_reg_idx;

    // registers used for bf16 conversions
    Xbyak::Zmm bf16_reg1;
    Xbyak::Zmm bf16_reg2;
    Xbyak::Zmm bf16_reg3;
    Xbyak::Reg64 bf16_reg4;
    Xbyak::Zmm bf16_reg5;
    Xbyak::Reg64 bf16_reg_mask;
    Xbyak::Opmask bf16_k_mask;
    int bf16_dq_reg_idx;
    const size_t bias_dt_size_;
    const size_t cstate_dt_size_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
