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

#ifndef CPU_GEMM_BF16_INNER_PRODUCT_HPP
#define CPU_GEMM_BF16_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "gemm/gemm.hpp"
#include "memory_tracking.hpp"
#include "jit_generator.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<data_type_t store_type>
struct gemm_bf16_ip_pp_kernel_t: public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_bf16_ip_pp_kernel_t<store_type>);
    /* row_length must be equal to OC for fwd w/ bias ip, for other case can be
       set to 0 to use the most appropriate unrolling */
    gemm_bf16_ip_pp_kernel_t(size_t row_length = 0,
        bool do_bias = false, bool do_relu = false);

    typedef typename prec_traits<store_type>::type store_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;

    void operator()(store_data_t *dst, const acc_data_t *acc,
            const acc_data_t *bias, float nslope,
            size_t start, size_t end);

    ~gemm_bf16_ip_pp_kernel_t() {
        delete bf16_emu_;
    }

private:
    struct ker_args {
        store_data_t *dst;
        const acc_data_t *acc;
        const acc_data_t *bias;
        float nslope;
        size_t len;
        size_t row_offset; // required for fwd ip w/ bias only
    };

    void (*ker_)(const ker_args *args);

    size_t row_length_;
    bool do_bias_;
    bool do_relu_;
    bool is_ajustable_row_length_;
    size_t def_unroll_, max_unroll_;
    size_t vlen_;

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_acc = rax;
    Xbyak::Reg64 reg_bias = rbx;

    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_row_offset = r9;
    Xbyak::Reg64 reg_rem_mask = r10;
    Xbyak::Opmask kreg_rem_mask = k1;
    Xbyak::Opmask kreg_relu_cmp = k2;

    Xbyak::Zmm vreg_zero = Xbyak::Zmm(0);
    Xbyak::Zmm vreg_nslope = Xbyak::Zmm(1);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(29);
    Xbyak::Reg64 bf16_emu_reserv_4 = r11;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(31);

    bool is_cpx_;
    bf16_emulation_t *bf16_emu_;

    void generate();
};

template <data_type_t dst_data_type>
struct gemm_bf16_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_inner_product_fwd_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && mayiuse(avx512_core)
                && this->set_default_params() == status::success
                && one_of(desc()->prop_kind, prop_kind::forward_training,
                        prop_kind::forward_inference)
                && !has_zero_dim_memory()
                && everyone_is(data_type::bf16,
                       desc()->src_desc.data_type,
                       desc()->weights_desc.data_type)
                && dst_data_type == desc()->dst_desc.data_type
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == desc()->bias_desc.data_type)
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_ == 1,
                        attr()->post_ops_.entry_[0].is_relu(true, false))
                && dense_gemm_consitency_check(src_pd(), weights_pd(),
                        dst_pd());
            if (!ok) return status::unimplemented;

            dst_is_acc_ = one_of(dst_data_type, data_type::f32);
            do_relu_ = attr()->post_ops_.len_ == 1
                && attr()->post_ops_.entry_[0].is_relu(true, false);

            init_scratchpad();

            return status::success;
        }

        bool dst_is_acc_;
        bool do_relu_;

    private:
        void init_scratchpad() {
            if (!dst_is_acc_) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                        sizeof(acc_data_t) * MB() * OC());
            }
        }
    };

    gemm_bf16_inner_product_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , pp_kernel_(nullptr)
    {
        bool do_relu = pd()->do_relu_;
        bool with_bias = pd()->with_bias();
        if (!pd()->dst_is_acc_ || do_relu || with_bias) {
            size_t row_size = with_bias ? pd()->OC() : 0;
            pp_kernel_ = new gemm_bf16_ip_pp_kernel_t<dst_data_type>(
                row_size, with_bias, do_relu);
        }
    }

    ~gemm_bf16_inner_product_fwd_t() {delete pp_kernel_;}

    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    gemm_bf16_ip_pp_kernel_t<dst_data_type> *pp_kernel_;
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t diff_src_data_type>
struct gemm_bf16_inner_product_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_inner_product_bwd_data_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && mayiuse(avx512_core)
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_data
                && !has_zero_dim_memory()
                && everyone_is(data_type::bf16,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && diff_src_data_type == desc()->diff_src_desc.data_type
                && attr()->has_default_values()
                && dense_gemm_consitency_check(diff_src_pd(), weights_pd(),
                        diff_dst_pd());
            if (!ok) return status::unimplemented;

            diff_src_is_acc_ = one_of(diff_src_data_type, data_type::f32);

            init_scratchpad();

            return status::success;
        }

        bool diff_src_is_acc_;

    private:
        void init_scratchpad() {
            if (!diff_src_is_acc_) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                        sizeof(acc_data_t) * MB() * IC_total_padded());
            }
        }
    };

    gemm_bf16_inner_product_bwd_data_t(const pd_t *apd,
            const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<diff_src_data_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t diff_wei_data_type>
struct gemm_bf16_inner_product_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR,
            gemm_bf16_inner_product_bwd_weights_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(avx512_core)
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_weights
                && !has_zero_dim_memory()
                && everyone_is(data_type::bf16,
                        desc()->src_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && diff_wei_data_type == desc()->diff_weights_desc.data_type
                && attr()->has_default_values()
                && dense_gemm_consitency_check(src_pd(), diff_weights_pd(),
                        diff_dst_pd());

            if (!ok) return status::unimplemented;

            diff_wei_is_acc_ = one_of(diff_wei_data_type, data_type::f32);

            init_scratchpad();

            return status::success;
        }

        bool diff_wei_is_acc_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            if (!diff_wei_is_acc_)
                scratchpad.book(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                        sizeof(acc_data_t) * OC() * IC_total_padded());

            if (with_bias())
                scratchpad.book(
                    memory_tracking::names::key_iprod_dst_bf16_convert_wsp,
                    sizeof(acc_data_t) * OC());
        }
    };

    gemm_bf16_inner_product_bwd_weights_t(const pd_t *apd,
            const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<diff_wei_data_type>::type diff_wei_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

