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

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
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
                && o_d.data_type() == data_type
                && o_d.is_dense();

            const auto n = src_pds_.size();
            for (size_t i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(&src_pds_[i]);
                ok = ok
                    && utils::everyone_is(data_type, i_d.data_type())
                    && i_d.format() == o_d.format()
                    && i_d.is_dense();
            }

            return ok ? success : unimplemented;
        }
    };

    simple_sum_t(const pd_t *conf, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*conf) {}

    virtual void execute(event_t *e) {
        execute();
        e->set_state(event_t::ready);
    }

    enum {max_num_arrs = 16 };
    typedef typename prec_traits<data_type>::type data_t;

private:
    void execute();
    pd_t conf_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
