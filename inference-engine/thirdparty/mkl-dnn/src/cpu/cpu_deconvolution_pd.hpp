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

#ifndef CPU_DECONVOLUTION_FWD_PD_HPP
#define CPU_DECONVOLUTION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "deconvolution_pd.hpp"
#include "convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define DECLARE_DECONVOLUTION_PD_t(...)                                        \
    virtual pd_t *clone() const override { return new pd_t(*this); }           \
    virtual status_t create_primitive(primitive_t **primitive,                 \
            const primitive_at_t *inputs, const primitive_t **outputs)         \
            const override {                                                   \
        double ms = get_msec();                                                \
        using namespace prop_kind;                                             \
        primitive_t::input_vector ins(inputs, inputs + this->n_inputs());      \
        primitive_t::output_vector outs(outputs, outputs + this->n_outputs()); \
        auto ret = safe_ptr_assign<primitive_t>(                               \
                *primitive, new (__VA_ARGS__)(this, ins, outs));               \
        primitive_t *conv_primitive;                                           \
        if (this->desc()->prop_kind == backward_weights) {                     \
            primitive_at_t conv_inputs[2];                                     \
            conv_inputs[0] = inputs[1];                                        \
            conv_inputs[1] = inputs[0];                                        \
            conv_pd_->create_primitive(                                        \
                    (&conv_primitive), conv_inputs, outputs);                  \
        } else                                                                 \
            conv_pd_->create_primitive((&conv_primitive), inputs, outputs);    \
        ((__VA_ARGS__ *)(*primitive))->conv_p_ = conv_primitive;               \
        ms = get_msec() - ms;                                                  \
        if (mkldnn_verbose()->level >= 2) {                                    \
            printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms);         \
            fflush(0);                                                         \
        }                                                                      \
        return ret;                                                            \
    }                                                                          \
    virtual const char *name() const override { return conv_pd_->name(); }

#define DECLARE_DECONVOLUTION_PD_T(...) DECLARE_DECONVOLUTION_PD_t(__VA_ARGS__)

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_deconvolution_fwd_pd_t: public deconvolution_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_deconvolution_fwd_pd_t(engine_t *engine,
            const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(this->engine_, &this->desc_.src_desc)
        , dst_pd_(this->engine_, &this->desc_.dst_desc)
        , weights_pd_(this->engine_, &this->desc_.weights_desc)
        , bias_pd_(this->engine_, &this->desc_.bias_desc) {}
    virtual ~cpu_deconvolution_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;
};

struct cpu_deconvolution_bwd_data_pd_t: public deconvolution_bwd_data_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_deconvolution_bwd_data_pd_t(engine_t *engine,
            const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_pd_(this->engine_, &this->desc_.diff_src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , weights_pd_(this->engine_, &this->desc_.weights_desc) {}
    virtual ~cpu_deconvolution_bwd_data_pd_t() {}

    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &weights_pd_ : nullptr; }

protected:
    cpu_memory_pd_t diff_src_pd_, diff_dst_pd_;
    cpu_memory_pd_t weights_pd_;
};

struct cpu_deconvolution_bwd_weights_pd_t: public deconvolution_bwd_weights_pd_t
{
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_deconvolution_bwd_weights_pd_t(engine_t *engine,
            const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(this->engine_, &this->desc_.src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , diff_weights_pd_(this->engine_, &this->desc_.diff_weights_desc)
        , diff_bias_pd_(this->engine_, &this->desc_.diff_bias_desc) {}
    virtual ~cpu_deconvolution_bwd_weights_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const override
    {
        if (index == 0) return &diff_weights_pd_;
        if (index == 1 && this->with_bias()) return &diff_bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_;
    cpu_memory_pd_t diff_dst_pd_;
    cpu_memory_pd_t diff_weights_pd_, diff_bias_pd_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
