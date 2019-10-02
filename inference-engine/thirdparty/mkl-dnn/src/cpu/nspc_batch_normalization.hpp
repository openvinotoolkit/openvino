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

#ifndef CPU_NSPC_BATCH_NORMALIZATION_HPP
#define CPU_NSPC_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_batch_normalization_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct nspc_batch_normalization_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(
                      engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nspc_bnorm:any", nspc_batch_normalization_fwd_t);

        virtual status_t init() override {
            using namespace data_type;
            using namespace prop_kind;

            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                    /* the algorithm requires barriers while switching
                     * between parallelization over N and C dimensions */
                    && mkldnn_thr_syncable() && is_fwd()
                    && !has_zero_dim_memory()
                    && utils::one_of(desc()->prop_kind, forward_training,
                               forward_inference)
                    && desc()->data_desc.data_type == data_type
                    && IMPLICATION(use_scaleshift(),
                               desc()->data_scaleshift_desc.data_type
                                       == f32)
                    && utils::everyone_is(f32, desc()->mean_desc.data_type,
                               desc()->variance_desc.data_type)
                    // TODO: add ndhwc support?
                    && utils::one_of(
                               data_pd_.desc()->format, memory_format::nhwc)
                    && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                    && (attr()->has_default_values()
                               || this->with_relu_post_op());
            if (!ok)
                return status::unimplemented;

            if (is_training() && fuse_bn_relu())
                bn_init_default_ws(this, this->workspace_pd_, 8);

            if (stats_is_src() || is_training()) {
                memory_desc_t stats_d;
                dims_t stats_dims = { C() };
                mkldnn_memory_desc_init(
                        &stats_d, 1, stats_dims, f32, memory_format::x);
                mean_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
                variance_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (!stats_is_src()) {
                const size_t stats_buf_sz = sizeof(acc_data_t)
                    * nstl::max(C(), 16) * mkldnn_get_max_threads();
                scratchpad.book(key_bnorm_reduction, stats_buf_sz);
                scratchpad.book(key_bnorm_tmp_mean, stats_buf_sz);
                scratchpad.book(key_bnorm_tmp_var, stats_buf_sz);
            }
            if (data_type == data_type::bf16) {
                const int simd_w = 16;
                const int nbufs = 2;
                const size_t bf16cvt_buf_sz = sizeof(acc_data_t) * nbufs
                    * mkldnn_get_max_threads() * utils::rnd_up(C(), simd_w);
                scratchpad.book(key_bnorm_bf16cvt, bf16cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<data_type>::type data_t;
    typedef float acc_data_t;

    nspc_batch_normalization_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    ~nspc_batch_normalization_fwd_t() {}

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t data_type>
struct nspc_batch_normalization_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(
                      engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nspc_bnorm:any", nspc_batch_normalization_bwd_t);

        virtual status_t init() override {
            using namespace data_type;
            using namespace prop_kind;

            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                    /* the algorithm requires barriers while switching
                     * between parallelization over N and C dimensions */
                    && mkldnn_thr_syncable() && is_bwd()
                    && !has_zero_dim_memory()
                    && utils::one_of(desc()->prop_kind, backward, backward_data)
                    && utils::everyone_is(data_type,
                               desc()->data_desc.data_type,
                               desc()->diff_data_desc.data_type)
                    && utils::everyone_is(f32, desc()->mean_desc.data_type,
                               desc()->variance_desc.data_type)
                    && IMPLICATION(use_scaleshift(),
                               desc()->diff_data_scaleshift_desc.data_type
                                               == f32
                                       && desc()->data_scaleshift_desc.data_type
                                               == f32)
                    && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                    // TODO: add ndhwc support?
                    && utils::one_of(
                               data_pd_.desc()->format, memory_format::nhwc)
                    && (attr()->has_default_values()
                               || this->with_relu_post_op())
                    && hint_fwd_pd_ != nullptr;
            if (!ok)
                return status::unimplemented;

            if (fuse_bn_relu()) {
                bn_init_default_ws(this, this->workspace_pd_, 8);
                const size_t this_ws_sz
                    = memory_desc_wrapper(this->workspace_pd()).size();

                bool ws_ok = true
                    && hint_fwd_pd_->workspace_pd()
                    && memory_desc_wrapper(hint_fwd_pd_->workspace_pd()).size()
                    == this_ws_sz;
                if (!ws_ok) return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_bnorm_reduction,
                    sizeof(acc_data_t) * 2 * C() * mkldnn_get_max_threads());
            scratchpad.book(key_bnorm_tmp_diff_ss, sizeof(acc_data_t) * 2 * C()
                    * (mkldnn_get_max_threads() + 1));
            if (data_type == data_type::bf16) {
                const int simd_w = 16;
                const int nbufs = 2 + !use_global_stats();
                const size_t bf16cvt_buf_sz = sizeof(acc_data_t) * nbufs
                    * mkldnn_get_max_threads() * utils::rnd_up(C(), simd_w);
                scratchpad.book(key_bnorm_bf16cvt, bf16cvt_buf_sz);
            }
        }
    };

    typedef typename prec_traits<data_type>::type data_t;
    typedef float acc_data_t;

    nspc_batch_normalization_bwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    ~nspc_batch_normalization_bwd_t() {}

    virtual void execute(event_t *e) const {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
