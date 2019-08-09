/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef SIMPLE_SUM_HPP
#define SIMPLE_SUM_HPP

#include "cpu_sum.hpp"
#include "cpu_isa_traits.hpp"
#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
struct sum_bf16_params_t {
    size_t ws_cvt_elements_per_thread_;
    size_t ws_acc_elements_per_thread_;
    size_t ws_elements_per_thread_;
    size_t acc_loop_step_;
};
}

template <data_type_t src_data_type, data_type_t dst_data_type>
struct simple_sum_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    struct pd_t: public cpu_sum_pd_t {
        pd_t(const memory_desc_t *output_d, int n, const float *scales,
             const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
            : cpu_sum_pd_t(output_d, n, scales, input_pds, attr) {}

        DECLARE_CPU_SUM_PD_T("simple:any", simple_sum_t);

        virtual status_t init() override {
            bool ok = true
                && cpu_sum_pd_t::init() == success
                && src_pds_.size() <= max_num_arrs;
            if (!ok) return unimplemented;

            const memory_desc_wrapper o_d(&dst_pd_);
            ok = ok
                && o_d.data_type() == dst_data_type
                && o_d.is_dense();
            if (!ok) return unimplemented;

            const auto n = src_pds_.size();
            for (size_t i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(&src_pds_[i]);
                ok = true
                    && utils::everyone_is(src_data_type, i_d.data_type())
                    && i_d.format() == o_d.format()
                    && i_d.is_dense();
                if (!ok) return unimplemented;
            }

            compute_blocking();
            init_scratchpad();

            return success;
        }

        sum_bf16_params_t bf16_p_;
        size_t block_size_, nelems_, blocks_number_, tail_;

        private:

            const size_t cacheline_size_ = 64; // bytes
            const size_t half_L1_size_ = 16 * 1024; // bytes

            void compute_blocking() {
                block_size_ = (src_data_type == data_type::bf16
                        ?  16 * cacheline_size_
                        : half_L1_size_)
                    / sizeof(src_data_type);
                nelems_ = memory_desc_wrapper(dst_pd()).nelems();
                blocks_number_ = nelems_ / block_size_;
                tail_ = nelems_ % block_size_;
            }

            void init_scratchpad() {
                if (src_data_type == data_type::bf16) {
                    bool is_dst_bf16_ = dst_data_type == data_type::bf16;
                    bf16_p_.ws_cvt_elements_per_thread_ =
                        cacheline_size_ / sizeof(acc_data_t);

                    bf16_p_.ws_acc_elements_per_thread_ =
                        is_dst_bf16_
                        ? bf16_p_.ws_cvt_elements_per_thread_
                        : 0;

                    bf16_p_.acc_loop_step_ = is_dst_bf16_
                        ? bf16_p_.ws_cvt_elements_per_thread_
                        : 1;

                    bf16_p_.ws_elements_per_thread_ = bf16_p_.ws_cvt_elements_per_thread_
                        + bf16_p_.ws_acc_elements_per_thread_;
                    size_t bf16cvt_buf_sz_ = sizeof(acc_data_t) * bf16_p_.ws_elements_per_thread_
                        * mkldnn_get_max_threads();
                    auto scratchpad = scratchpad_registry().registrar();
                    scratchpad.book(memory_tracking::names::key_sum_bf16cvt, bf16cvt_buf_sz_);
                }
            }
    };

    simple_sum_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
    }

    virtual void execute(event_t *e) const {
        execute();
        e->set_state(event_t::ready);
    }

    enum {max_num_arrs = 16 };
    typedef typename prec_traits<src_data_type>::type src_data_t;
    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;

private:
    void execute() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
