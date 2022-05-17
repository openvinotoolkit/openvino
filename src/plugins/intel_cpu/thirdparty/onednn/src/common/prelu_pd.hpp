/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef COMMON_PRELU_PD_HPP
#define COMMON_PRELU_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive_desc.hpp"

namespace dnnl {
namespace impl {

struct prelu_fwd_pd_t;

struct prelu_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::prelu;

    const prelu_desc_t *desc() const { return &desc_; }

    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::prelu_d: *(const prelu_desc_t **)result = desc(); break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    dim_t N() const { return data_desc().dims[0]; }
    dim_t C() const { return data_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? data_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_desc().dims[ndims() - 1] : 1; }

    int ndims() const { return data_desc().ndims; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(desc_.data_desc).has_zero_dim();
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    const memory_desc_t *weights_md(int index) const override {
        return index == 0 ? &weights_md_ : &glob_zero_md;
    }

    const memory_desc_t *src_md(int index) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }

    const memory_desc_t *dst_md(int index) const override {
        return index == 0 ? &data_md_ : &glob_zero_md;
    }

    size_t dtype_size() const {
        return types::data_type_size(data_md_.data_type);
    }

protected:
    prelu_desc_t desc_;
    const prelu_fwd_pd_t *hint_fwd_pd_;
    memory_desc_t data_md_;
    memory_desc_t weights_md_;

    prelu_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , data_md_(desc_.data_desc)
        , weights_md_(desc_.weights_desc) {}

private:
    const memory_desc_t &data_desc() const { return desc_.data_desc; }
};

struct prelu_fwd_pd_t : public prelu_pd_t {
    typedef prelu_fwd_pd_t base_class;
    typedef prelu_fwd_pd_t hint_class;

    primitive_desc_t::arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
        if (arg == DNNL_ARG_WEIGHTS) return arg_usage_t::input;
        if (arg == DNNL_ARG_DST) return arg_usage_t::output;
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_DST: return dst_md(0);
            default: return prelu_pd_t::arg_md(arg);
        }
    }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }

protected:
    prelu_fwd_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd)
        : prelu_pd_t(adesc, attr, hint_fwd_pd) {}

    bool set_default_formats() {
        if (weights_md_.format_kind == format_kind::any)
            if (memory_desc_init_by_blocking_desc(
                        weights_md_, data_md_.format_desc.blocking)
                    != status::success)
                return false;
        return true;
    }
};

struct prelu_bwd_pd_t : public prelu_pd_t {
    typedef prelu_bwd_pd_t base_class;
    typedef prelu_fwd_pd_t hint_class;

    primitive_desc_t::arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
        if (arg == DNNL_ARG_WEIGHTS) return arg_usage_t::input;
        if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;
        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;
        if (arg == DNNL_ARG_DIFF_WEIGHTS) return arg_usage_t::output;
        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
            case DNNL_ARG_DIFF_WEIGHTS: return diff_weights_md(0);
            default: return prelu_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *diff_src_md(int index) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }

    const memory_desc_t *diff_dst_md(int index) const override {
        return index == 0 ? &diff_data_md_ : &glob_zero_md;
    }

    const memory_desc_t *diff_weights_md(int index) const override {
        return index == 0 ? &diff_weights_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 3; }
    int n_outputs() const override { return 2; }

protected:
    memory_desc_t diff_data_md_;
    memory_desc_t diff_weights_md_;

    prelu_bwd_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd)
        : prelu_pd_t(adesc, attr, hint_fwd_pd)
        , diff_data_md_(desc_.diff_data_desc)
        , diff_weights_md_(desc_.diff_weights_desc) {}

    bool set_default_formats() {
        if (weights_md_.format_kind == format_kind::any)
            if (memory_desc_init_by_blocking_desc(
                        weights_md_, data_md_.format_desc.blocking)
                    != status::success)
                return false;
        if (diff_weights_md_.format_kind == format_kind::any)
            if (memory_desc_init_by_blocking_desc(
                        diff_weights_md_, data_md_.format_desc.blocking)
                    != status::success)
                return false;
        return true;
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
