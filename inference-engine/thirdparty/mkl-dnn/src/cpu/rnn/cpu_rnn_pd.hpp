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

#ifndef CPU_RNN_PD_HPP
#define CPU_RNN_PD_HPP

#include "c_types_map.hpp"
#include "../cpu_engine.hpp"
#include "../cpu_memory.hpp"
#include "../cpu_primitive.hpp"
#include "nstl.hpp"
#include "rnn_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_rnn_fwd_pd_t : public rnn_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_rnn_fwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const primitive_attr_t *attr, const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_layer_pd_(engine, &desc_.src_layer_desc)
        , src_iter_pd_(engine, &desc_.src_iter_desc)
        , weights_layer_pd_(engine, &desc_.weights_layer_desc)
        , weights_iter_pd_(engine, &desc_.weights_iter_desc)
        , bias_pd_(engine, &desc_.bias_desc)
        , dst_layer_pd_(engine, &desc_.dst_layer_desc)
        , dst_iter_pd_(engine, &desc_.dst_iter_desc)
        , ws_pd_(engine_) {}
    virtual ~cpu_rnn_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
        if (index == 0)
            return &src_layer_pd_;
        if (index == 1 && this->with_src_iter())
            return &src_iter_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0)
            return &weights_layer_pd_;
        if (index == 1)
            return &weights_iter_pd_;
        if (index == 2 && this->with_bias())
            return &bias_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override {
        if (index == 0)
            return &dst_layer_pd_;
        if (index == 1 && this->with_dst_iter())
            return &dst_iter_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override {
        return (index == 0 && !ws_pd_.is_zero()) ? &ws_pd_ : nullptr;
    }

protected:
    cpu_memory_pd_t src_layer_pd_;
    cpu_memory_pd_t src_iter_pd_;
    cpu_memory_pd_t weights_layer_pd_;
    cpu_memory_pd_t weights_iter_pd_;
    cpu_memory_pd_t bias_pd_;
    cpu_memory_pd_t dst_layer_pd_;
    cpu_memory_pd_t dst_iter_pd_;
    cpu_memory_pd_t ws_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_layer_pd_.desc()->format == any)
            CHECK(src_layer_pd_.set_format(tnc));
        if (dst_layer_pd_.desc()->format == any)
            CHECK(dst_layer_pd_.set_format(tnc));

        // Optional parameters
        if ((!src_iter_pd_.is_zero()) && (src_iter_pd_.desc()->format == any))
            CHECK(src_iter_pd_.set_format(ldsnc));
        if ((!bias_pd_.is_zero()) && (bias_pd_.desc()->format == any))
            CHECK(bias_pd_.set_format(ldgo));
        if ((!dst_iter_pd_.is_zero()) && (dst_iter_pd_.desc()->format == any))
            CHECK(dst_iter_pd_.set_format(ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace memory_format;
        using namespace utils;
        using namespace data_type;
        bool ok = true;
        ok = ok && src_layer_pd_.desc()->format == tnc
                && dst_layer_pd_.desc()->format == tnc;
        ok = ok && IMPLICATION(!src_iter_pd_.is_zero(),
                           src_iter_pd_.desc()->format == ldsnc)
                && IMPLICATION(!dst_iter_pd_.is_zero(),
                           dst_iter_pd_.desc()->format == ldsnc);

        ok = ok && one_of(weights_layer_pd_.desc()->format, ldigo, rnn_packed)
                && one_of(weights_iter_pd_.desc()->format, ldigo, rnn_packed);
        ok = ok && IMPLICATION(weights_iter_pd_.desc()->format == rnn_packed,
                           weights_iter_pd_.desc()
                                           ->layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldigo_p);
        ok = ok && IMPLICATION(weights_layer_pd_.desc()->format == rnn_packed,
                           weights_layer_pd_.desc()
                                           ->layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldigo_p);

        ok = ok && IMPLICATION(!bias_pd_.is_zero(),
                           bias_pd_.desc()->format == ldgo);

        /* Int8 is supported only for packed weights */
        data_type_t weights_iter_dt = weights_iter_pd_.desc()->data_type;
        data_type_t weights_layer_dt = weights_layer_pd_.desc()->data_type;
        ok = ok && IMPLICATION(weights_iter_dt == s8,
                           weights_iter_pd_.desc()->format == rnn_packed);
        ok = ok && IMPLICATION(weights_layer_dt == s8,
                           weights_layer_pd_.desc()->format == rnn_packed);

        return ok ? status::success : status::unimplemented;
    }
};

struct cpu_rnn_bwd_pd_t : public rnn_bwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_rnn_bwd_pd_t(engine_t *engine, const rnn_desc_t *adesc,
            const primitive_attr_t *attr, const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_layer_pd_(engine, &desc_.src_layer_desc)
        , src_iter_pd_(engine, &desc_.src_iter_desc)
        , weights_layer_pd_(engine, &desc_.weights_layer_desc)
        , weights_iter_pd_(engine, &desc_.weights_iter_desc)
        , bias_pd_(engine, &desc_.bias_desc)
        , dst_layer_pd_(engine, &desc_.dst_layer_desc)
        , dst_iter_pd_(engine, &desc_.dst_iter_desc)
        , diff_src_layer_pd_(engine, &desc_.diff_src_layer_desc)
        , diff_states_pd_(engine, &desc_.diff_src_iter_desc)
        , diff_weights_layer_pd_(engine, &desc_.diff_weights_layer_desc)
        , diff_weights_iter_pd_(engine, &desc_.diff_weights_iter_desc)
        , diff_bias_pd_(engine, &desc_.diff_bias_desc)
        , diff_dst_layer_pd_(engine, &desc_.diff_dst_layer_desc)
        , diff_dst_iter_pd_(engine, &desc_.diff_dst_iter_desc)
        , ws_pd_(engine_) {}
    virtual ~cpu_rnn_bwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
        if (index == 0)
            return &src_layer_pd_;
        if (index == 1 && this->with_src_iter())
            return &src_iter_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0)
            return &weights_layer_pd_;
        if (index == 1)
            return &weights_iter_pd_;
        if (index == 2 && this->with_bias())
            return &bias_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override {
        if (index == 0)
            return &dst_layer_pd_;
        if (index == 1 && this->with_dst_iter())
            return &dst_iter_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override {
        if (index == 0)
            return &diff_src_layer_pd_;
        if (index == 1 && this->with_src_iter())
            return &diff_states_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *diff_weights_pd(
            int index = 0) const override {
        if (index == 0)
            return &diff_weights_layer_pd_;
        if (index == 1)
            return &diff_weights_iter_pd_;
        if (index == 2 && this->with_bias())
            return &diff_bias_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override {
        if (index == 0)
            return &diff_dst_layer_pd_;
        if (index == 1 && this->with_dst_iter())
            return &diff_dst_iter_pd_;
        return nullptr;
    }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override {
        return (index == 0 && !ws_pd_.is_zero()) ? &ws_pd_ : nullptr;
    }

protected:
    cpu_memory_pd_t src_layer_pd_;
    cpu_memory_pd_t src_iter_pd_;
    cpu_memory_pd_t weights_layer_pd_;
    cpu_memory_pd_t weights_iter_pd_;
    cpu_memory_pd_t bias_pd_;
    cpu_memory_pd_t dst_layer_pd_;
    cpu_memory_pd_t dst_iter_pd_;
    cpu_memory_pd_t diff_src_layer_pd_;
    cpu_memory_pd_t diff_states_pd_;
    cpu_memory_pd_t diff_weights_layer_pd_;
    cpu_memory_pd_t diff_weights_iter_pd_;
    cpu_memory_pd_t diff_bias_pd_;
    cpu_memory_pd_t diff_dst_layer_pd_;
    cpu_memory_pd_t diff_dst_iter_pd_;
    cpu_memory_pd_t ws_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_layer_pd_.desc()->format == any)
            CHECK(src_layer_pd_.set_format(tnc));
        if (diff_src_layer_pd_.desc()->format == any)
            CHECK(diff_src_layer_pd_.set_format(tnc));
        if (diff_weights_layer_pd_.desc()->format == any) {
            memory_desc_t md = *(diff_weights_layer_pd_.desc());
            md.format = ldigo;
            CHECK(memory_desc_wrapper::compute_blocking(md));
            CHECK(rnn_utils::set_good_strides(md));
            cpu_memory_t::pd_t new_pd(engine_, &md);
            diff_weights_layer_pd_ = new_pd;
        }
        if (diff_weights_iter_pd_.desc()->format == any) {
            memory_desc_t md = *(diff_weights_iter_pd_.desc());
            md.format = ldigo;
            CHECK(memory_desc_wrapper::compute_blocking(md));
            CHECK(rnn_utils::set_good_strides(md));
            cpu_memory_t::pd_t new_pd(engine_, &md);
            diff_weights_iter_pd_ = new_pd;
        }
        if (dst_layer_pd_.desc()->format == any)
            CHECK(dst_layer_pd_.set_format(tnc));
        if (diff_dst_layer_pd_.desc()->format == any)
            CHECK(diff_dst_layer_pd_.set_format(tnc));

        // Optional parameters
        if ((!src_iter_pd_.is_zero()) && (src_iter_pd_.desc()->format == any))
            CHECK(src_iter_pd_.set_format(ldsnc));
        if ((!diff_states_pd_.is_zero())
                && (diff_states_pd_.desc()->format == any))
            CHECK(diff_states_pd_.set_format(ldsnc));
        if ((!bias_pd_.is_zero()) && (bias_pd_.desc()->format == any))
            CHECK(bias_pd_.set_format(ldgo));
        if ((!diff_bias_pd_.is_zero()) && (diff_bias_pd_.desc()->format == any))
            CHECK(diff_bias_pd_.set_format(ldgo));
        if ((!dst_iter_pd_.is_zero()) && (dst_iter_pd_.desc()->format == any))
            CHECK(dst_iter_pd_.set_format(ldsnc));
        if ((!diff_dst_iter_pd_.is_zero())
                && (diff_dst_iter_pd_.desc()->format == any))
            CHECK(diff_dst_iter_pd_.set_format(ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace memory_format;
        using namespace utils;
        bool ok = true;
        ok = ok && src_layer_pd_.desc()->format == tnc
                && dst_layer_pd_.desc()->format == tnc;
        ok = ok && IMPLICATION(!src_iter_pd_.is_zero(),
                           src_iter_pd_.desc()->format == ldsnc)
                && IMPLICATION(!dst_iter_pd_.is_zero(),
                           dst_iter_pd_.desc()->format == ldsnc);

        ok = ok && one_of(weights_layer_pd_.desc()->format, ldgoi, rnn_packed)
                && one_of(weights_iter_pd_.desc()->format, ldgoi, rnn_packed);
        ok = ok && IMPLICATION(weights_iter_pd_.desc()->format == rnn_packed,
                           weights_iter_pd_.desc()
                                           ->layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldgoi_p);
        ok = ok && IMPLICATION(weights_layer_pd_.desc()->format == rnn_packed,
                           weights_layer_pd_.desc()
                                           ->layout_desc.rnn_packed_desc.format
                                   == mkldnn_ldgoi_p);

        ok = ok && IMPLICATION(!bias_pd_.is_zero(),
                           bias_pd_.desc()->format == ldgo);

        ok = ok && diff_src_layer_pd_.desc()->format == tnc
                && diff_dst_layer_pd_.desc()->format == tnc;
        ok = ok && IMPLICATION(!diff_states_pd_.is_zero(),
                           diff_states_pd_.desc()->format == ldsnc)
                && IMPLICATION(!diff_dst_iter_pd_.is_zero(),
                           diff_dst_iter_pd_.desc()->format == ldsnc);
        ok = ok && diff_weights_layer_pd_.desc()->format == ldigo
                && diff_weights_iter_pd_.desc()->format == ldigo;
        ok = ok && IMPLICATION(!diff_bias_pd_.is_zero(),
                           diff_bias_pd_.desc()->format == ldgo);

        return ok ? status::success : status::unimplemented;
    }
};
}
}
}

#endif
