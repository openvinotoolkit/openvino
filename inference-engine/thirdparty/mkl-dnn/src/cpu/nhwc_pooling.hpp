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

#ifndef CPU_NHWC_POOLING_HPP
#define CPU_NHWC_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "cpu_pooling_pd.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace nhwc_pooling {
size_t strided_offset(const int _n, const size_t _sn, const int _d,
        const size_t _sd, const int _h, const size_t _sh, const int _w,
        const size_t _sw);
}

template <impl::data_type_t data_type>
struct nhwc_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nhwc_pooling:any", nhwc_pooling_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace memory_format;
            assert(engine()->kind() == engine_kind::cpu);
            auto src_format = src_pd()->desc()->format;
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && utils::everyone_is(data_type,
                        src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && utils::one_of(src_format, nhwc, ndhwc)
                && (src_format == dst_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                // Allocate dense workspace buffer based on logical dimensions
                // of the output dst
                memory_desc_t indices_desc;
                if (is_3d()) {
                    dims_t ws_dims = { MB(), C(), OD(), OH(), OW() };
                    mkldnn_memory_desc_init(&indices_desc, 5, ws_dims,
                            pooling_index_data_type(desc()),
                            memory_format::ndhwc);
                } else {
                    dims_t ws_dims = { MB(), C(), OH(), OW() };
                    mkldnn_memory_desc_init(&indices_desc, 4, ws_dims,
                            pooling_index_data_type(desc()),
                            memory_format::nhwc);
                }
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return status::success;
        }
    };

    nhwc_pooling_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}

    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    void array_div_by_const(const int n, const data_t *src, const size_t num,
            data_t *dst);
    void array_add(const int n, const data_t *src, data_t *dst);

    template <bool use_workspace>
    void array_nhwc_max(const int n, data_t *dst, const data_t *src,
            unsigned char *ws, const size_t ws_offset, const data_type_t ws_dt,
            const int index) {
        assert(!((use_workspace == false) ^ (!ws))); // ensure ws pointer exists
        PRAGMA_OMP_SIMD()
        for (int oc = 0; oc < n; ++oc) {
            auto s = src[oc];
            data_t mv = dst[oc];

            // update index of maximum
#if defined __INTEL_COMPILER
            if ((use_workspace) && (s > mv)) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
                if (ws_dt == data_type::u8) {
                    assert(0 <= index && index <= 255);
                    ws[ws_offset + oc] = index;
                } else
                    reinterpret_cast<int *>(ws)[ws_offset + oc] = index;
            }
#else
            // Need to add explicit predicates for GCC to vectorize this.
            // And although the resulting code is ugly, it is still 4 times
            // faster than scalar
            if (use_workspace) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

                if (ws_dt == data_type::u8) {
                    assert(0 <= index && index <= 255);
                    unsigned char predicate = (s > mv) ? 0xff : 0;
                    unsigned char current_value = ws[ws_offset + oc];
                    current_value = (predicate & (unsigned char)index)
                        | ((~predicate) & current_value);
                    ws[ws_offset + oc] = current_value;
                } else {
                    auto wint = reinterpret_cast<int *>(ws);
                    unsigned int predicate = (s > mv) ? 0xffffffff : 0;
                    unsigned int current_value = wint[ws_offset + oc];
                    current_value = (predicate & (unsigned int)index)
                        | ((~predicate) & current_value);
                    wint[ws_offset + oc] = current_value;
                }
            }
#endif
            // update maximum
            dst[oc] = nstl::max(s, mv);
        }
    }

    template <bool use_workspace>
    void array_nhwc_initialize(const int n, data_t *dst, unsigned char *ws,
            const size_t ws_offset, const data_type_t ws_dt) {
        assert(!((use_workspace == false) ^ (!ws))); // ensure ws pointer exists
        for (int oc = 0; oc < n; ++oc) {
            if (use_workspace) {
                assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
                if (ws_dt == data_type::u8) {
                    ws[ws_offset + oc] = 0;
                } else
                    reinterpret_cast<int *>(ws)[ws_offset + oc] = 0;
            }
            dst[oc] = nstl::numeric_limits<data_t>::lowest();
        }
    }

    pd_t conf_;
};

template <impl::data_type_t data_type>
struct nhwc_pooling_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_bwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nhwc:any", nhwc_pooling_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace memory_format;
            assert(engine()->kind() == engine_kind::cpu);
            auto diff_dst_format = diff_dst_pd()->desc()->format;
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, backward_data)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && utils::everyone_is(data_type,
                        diff_dst_pd()->desc()->data_type,
                        diff_src_pd()->desc()->data_type)
                && utils::one_of(diff_dst_format, nhwc, ndhwc)
                && (diff_dst_format == diff_src_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                bool ws_ok = true
                    && hint_fwd_pd_
                    && hint_fwd_pd_->workspace_pd()
                    && utils::one_of(
                            hint_fwd_pd_->workspace_pd()->desc()->format,
                            nhwc, ndhwc)
                    && hint_fwd_pd_->workspace_pd()->engine()->kind()
                            == engine_kind::cpu;
                if (!ws_ok) return status::unimplemented;

                ws_pd_ = *(cpu_memory_t::pd_t *)hint_fwd_pd_->workspace_pd();
            }

            return status::success;
        }
    };

    nhwc_pooling_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
};

}// namespace cpu
}// namespace impl
}// namespace mkldnn

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
