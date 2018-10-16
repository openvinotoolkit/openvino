/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_REF_SOFTMAX_FWD_HPP
#define CPU_REF_SOFTMAX_FWD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_softmax_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_softmax_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_softmax_fwd_pd_t {
        pd_t(engine_t *engine, const softmax_desc_t *adesc,
                const primitive_attr_t *attr,
                const softmax_fwd_pd_t *hint_fwd_pd)
            : cpu_softmax_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(desc()->prop_kind, forward_inference,
                        forward_training)
                && data_pd_.desc()->data_type == data_type
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_softmax_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), ws_(nullptr) {
        auto ndims = conf_.desc()->data_desc.ndims;
        auto dims = conf_.desc()->data_desc.dims;
        auto axis = conf_.desc()->softmax_axis;

        outer_size_ = utils::array_product(dims, axis);
        channels_ = dims[axis];
        inner_size_ = utils::array_product(dims + axis + 1, ndims - axis - 1);
        val_max_ = val_denom_ = 0;

        if (inner_size_ > 1) {
            ws_ = new data_t[2*inner_size_];
            max_ = &ws_[0];
            denom_ = &ws_[inner_size_];
        } else {
            max_ = &val_max_;
            denom_ = &val_denom_;
        }

        const memory_desc_wrapper data_d(conf_.src_pd());
        use_dense_ = inner_size_ == 1 && data_d.is_dense()
            && data_d.blocking_desc().block_dims[axis] == 1
            && data_d.blocking_desc().strides[0][axis] == 1;
    }
    ~ref_softmax_fwd_t() { if (ws_) delete [] ws_; }
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (use_dense_) execute_forward_dense();
        else execute_forward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward_dense();
    void execute_forward_generic();

    void _max(int n, const data_t *x, data_t *max_data);
    void _sub(int n, data_t alpha, const data_t *x, data_t *y);
    void _exp(int n, const data_t *a, data_t *r);
    void _sum(int n, const data_t *x, data_t *sum_data);
    void _scal(int n, data_t alpha, data_t *x);

    pd_t conf_;

    bool use_dense_;
    int outer_size_, channels_, inner_size_;
    data_t val_max_, val_denom_;
    data_t *ws_, *max_, *denom_;
};

template <impl::data_type_t data_type>
struct ref_softmax_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_softmax_bwd_pd_t {
        pd_t(engine_t *engine, const softmax_desc_t *adesc,
                const primitive_attr_t *attr,
                const softmax_fwd_pd_t *hint_fwd_pd)
            : cpu_softmax_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(desc()->prop_kind, backward_data)
                && diff_src_pd_.desc()->data_type == data_type
                && diff_dst_pd_.desc()->data_type == data_type
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_softmax_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
        auto dims = conf_.desc()->diff_desc.dims;
        auto axis = conf_.desc()->softmax_axis;
        auto ndims = conf_.desc()->diff_desc.ndims;

        outer_size_ = utils::array_product(dims, axis);
        channels_ = dims[axis];
        inner_size_ = utils::array_product(dims + axis + 1, ndims - axis - 1);

        // Diff desc as well as data desc whould be checked
        const memory_desc_wrapper data_d(conf_.dst_pd());
        const memory_desc_wrapper diff_d(conf_.diff_dst_pd());
        use_dense_ = true
            && inner_size_ == 1
            && diff_d == data_d
            && diff_d.is_dense()
            && diff_d.blocking_desc().block_dims[axis] == 1
            && diff_d.blocking_desc().strides[0][axis] == 1;
    }
    ~ref_softmax_bwd_t() {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (use_dense_) execute_backward_dense();
        else execute_backward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_dense();
    void execute_backward_generic();

    pd_t conf_;
    bool use_dense_;
    int outer_size_, channels_, inner_size_;
    data_t val_max_, val_denom_;
    data_t *max_, *denom_;
};


}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
