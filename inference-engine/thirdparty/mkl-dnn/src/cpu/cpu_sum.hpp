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

#ifndef CPU_SUM_HPP
#define CPU_SUM_HPP

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define DECLARE_CPU_SUM_PD_t(impl_name, ...) \
    static status_t create(sum_pd_t **sum_pd, \
            const memory_desc_t *output_d, int n, const float *scales, \
            const memory_pd_t **input_pds, const primitive_attr_t *attr) { \
        auto _pd = new pd_t(output_d, n, scales, \
                (const cpu_memory_pd_t **)input_pds, attr); \
        if (_pd == nullptr) return out_of_memory; \
        if (_pd->init() != success) { delete _pd; return unimplemented; } \
        return safe_ptr_assign<sum_pd_t>(*sum_pd, _pd); \
    } \
    virtual status_t create_primitive(primitive_t **primitive, \
            const primitive_at_t *inputs, \
            const primitive_t **outputs) const override { \
        double ms = get_msec(); \
        primitive_t::input_vector ins(inputs, inputs + n_); \
        primitive_t::output_vector outs(outputs, outputs + 1); \
        auto ret = safe_ptr_assign<primitive_t>(*primitive, \
                new (__VA_ARGS__)(this, ins, outs)); \
        ms = get_msec() - ms; \
        if (mkldnn_verbose()->level >= 2) { \
            printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms); \
            fflush(0); \
        } \
        return ret; \
    } \
    virtual pd_t *clone() const override { return new pd_t(*this); } \
    virtual const char *name() const override { return impl_name; }
#define DECLARE_CPU_SUM_PD_T(impl_name, ...) \
    DECLARE_CPU_SUM_PD_t(impl_name, __VA_ARGS__)

struct cpu_sum_pd_t: public sum_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_sum_pd_t(const memory_desc_t *output_d, int n, const float *scales,
            const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
        : sum_pd_t(input_pds[0]->engine(), n, attr),
        dst_pd_(input_pds[0]->engine()) {
        for (int i = 0; i < n_; ++i) {
            src_pds_.push_back(*input_pds[i]);
            scales_.push_back((float)scales[i]);
        }
        dst_pd_ = cpu_memory_pd_t(input_pds[0]->engine(), output_d);
    }

    virtual const cpu_memory_t::pd_t *src_pd(int index = 0) const override
    { return index < this->n_ ? &src_pds_[index] : nullptr; }
    virtual const cpu_memory_t::pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }

    nstl::vector<float> scales_;
protected:
    nstl::vector<cpu_memory_t::pd_t> src_pds_;
    cpu_memory_t::pd_t dst_pd_;

    virtual status_t init() {
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_pd(&src_pds_[i]);
            if (!src_pd.is_blocking_desc())
                return unimplemented;
        }
        bool ok = true
            && set_default_params() == success
            && attr()->has_default_values();
        return ok ? success : unimplemented;
    }

    virtual status_t set_default_params() {
        auto dst_fmt = dst_pd_.desc()->format;
        if (dst_fmt == memory_format::any) {
            /* the stupidest ever heuristics */
            for (int i = 0; i < n_; ++i)
                dst_fmt = nstl::max(dst_fmt, src_pds_[i].desc()->format);

            if (dst_fmt == memory_format::blocked)
                dst_pd_ = src_pds_[0];
            else
                CHECK(dst_pd_.set_format(dst_fmt));
        }

        return success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
