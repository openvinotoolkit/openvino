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

#ifndef RNN_PD_HPP
#define RNN_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "primitive_desc.hpp"

namespace mkldnn {
namespace impl {

// struct rnn_fwd_pd_t;

struct rnn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_pd_t(mkldnn::impl::engine_t *engine, const rnn_desc_t *adesc,
            const primitive_attr_t *attr, const rnn_pd_t *hint_pd)
        : primitive_desc_t(engine, attr, primitive_kind::rnn)
        , desc_(*adesc)
        , hint_pd_(hint_pd) {}
    virtual ~rnn_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    virtual void init_info() override { init_info_rnn(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::rnn_d: *(const rnn_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    inline bool is_training() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::backward);
    }

    inline bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    int T() const { return desc_.src_layer_desc.dims[0]; }
    int MB() const { return desc_.src_layer_desc.dims[1]; }

    int L() const { return desc_.weights_layer_desc.dims[0]; }
    int D() const { return desc_.weights_layer_desc.dims[1]; }

    int SIC() const { return desc_.weights_iter_desc.dims[2]; }

    int SLC() const { return desc_.weights_layer_desc.dims[2]; }
    int G() const { return desc_.weights_layer_desc.dims[3]; }
    int DIC() const { return desc_.weights_layer_desc.dims[4]; }

    int DLC() const { return desc_.dst_layer_desc.dims[2]; }

    bool with_bias() const {
        return !memory_desc_wrapper(desc_.bias_desc).is_zero();
    }

    bool with_src_iter() const {
        return !(memory_desc_wrapper(desc_.src_iter_desc).is_zero());
    }

    bool with_dst_iter() const {
        return !memory_desc_wrapper(desc_.dst_iter_desc).is_zero();
    }

    mkldnn::impl::alg_kind_t cell_kind() const {
        return desc_.cell_desc.cell_kind;
    }
    mkldnn::impl::alg_kind_t activation_kind() const {
        return desc_.cell_desc.activation_kind;
    }

    bool is_lbr() const {
        return cell_kind() == mkldnn_gru_linear_before_reset;
    }

    mkldnn_rnn_direction_t direction() const { return desc_.direction; }

protected:
    rnn_desc_t desc_;
    const rnn_pd_t *hint_pd_;
};

struct rnn_fwd_pd_t : public rnn_pd_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    using rnn_pd_t::rnn_pd_t;
    virtual ~rnn_fwd_pd_t() {}

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (index == 0) return src_pd(0);
        if (with_src_iter() && index == 1) return src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 3) return weights_pd(index);

        return nullptr;
    }

    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return dst_pd(0);
        if (with_dst_iter() && index == 1) return dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (is_training() && index == 0) return workspace_pd();

        return nullptr;
    }

    virtual int n_inputs() const override {
        return 3 + with_bias() + with_src_iter();
    }

    virtual int n_outputs() const override {
        return 1 + with_dst_iter() + is_training();
    }

    int ws_idx() const { return 1 + with_dst_iter(); }
};

struct rnn_bwd_pd_t : public rnn_pd_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    using rnn_pd_t::rnn_pd_t;
    virtual ~rnn_bwd_pd_t() {}

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (index == 0) return src_pd(0);
        if (with_src_iter() && index == 1) return src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 2) return weights_pd(index);
        if (with_bias() && index == 2) return weights_pd(2);
        index = index - 2 - with_bias();

        if (index == 0) return dst_pd(0);
        if (with_dst_iter() && index == 1) return dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (index == 0) return diff_dst_pd(0);
        if (with_dst_iter() && index == 1) return diff_dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (index == 0) return workspace_pd();

        return nullptr;
    }

    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return diff_src_pd(0);
        if (with_src_iter() && index == 1) return diff_src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 3) return diff_weights_pd(index);

        return nullptr;
    }

    virtual int n_inputs() const override {
        return 6 + with_src_iter() + with_bias() + 2 * with_dst_iter();
    }
    virtual int n_outputs() const override {
        return 3 + with_src_iter() + with_bias();
    }

    int ws_idx() const {
        return 5 + with_src_iter() + with_bias() + 2 * with_dst_iter();
    }
};
}
}

#endif
