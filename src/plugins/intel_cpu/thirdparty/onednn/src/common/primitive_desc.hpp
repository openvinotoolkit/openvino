/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_DESC_HPP
#define COMMON_PRIMITIVE_DESC_HPP

#include <typeindex>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "dnnl_sel_build.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "primitive_attr.hpp"
#include "primitive_cache.hpp"
#include "type_helpers.hpp"
#include "verbose.hpp"

namespace dnnl {
namespace impl {

struct impl_list_item_t;
struct primitive_t;
// Primitive descriptor implementation
struct primitive_desc_t : public c_compatible {
    primitive_desc_t(const primitive_attr_t *attr, primitive_kind_t kind)
        : attr_(*attr), kind_(kind), pd_iterator_offset_(0) {
        is_initialized_ = is_initialized_ && attr_.is_initialized();
    }

    primitive_desc_t(primitive_kind_t kind) : kind_(kind) {}

    bool is_initialized() const { return is_initialized_; }

    virtual ~primitive_desc_t() = default;
    virtual primitive_desc_t *clone() const = 0;

    const primitive_attr_t *attr() const { return &attr_; }
    primitive_kind_t kind() const { return kind_; }

    const char *info(engine_t *engine) const {
        if (!info_.is_initialized()) info_.init(engine, this);
        return info_.c_str();
    }

    memory_tracking::registry_t &scratchpad_registry() {
        return scratchpad_registry_;
    }
    const memory_tracking::registry_t &scratchpad_registry() const {
        return scratchpad_registry_;
    }

    virtual const op_desc_t *op_desc() const { return nullptr; }

    static bool post_op_has_proper_input(const primitive_attr_t *attr,
            const primitive_kind_t prim, const int idx, const int arg,
            const int src_mnemonic) {
        return (attr->post_ops_.contain(prim, idx)
                && arg == (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | src_mnemonic));
    }

    enum class arg_usage_t { unused, input, output };
    virtual arg_usage_t arg_usage(int arg) const {
        using types::is_zero_md;
        if (arg == DNNL_ARG_ATTR_OUTPUT_SCALES
                && !attr()->output_scales_.defined())
            return arg_usage_t::input;
        if ((arg & DNNL_ARG_ATTR_ZERO_POINTS)
                && !attr()->zero_points_.defined(arg))
            return arg_usage_t::input;
        if ((arg & (DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC))
            && !attr()->input_zero_points_.has_default_values())
            return arg_usage_t::input;
        if ((arg & (DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS))
            && !attr()->weights_zero_points_.has_default_values())
            return arg_usage_t::input;
        if ((arg & (DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST))
            && !attr()->output_compensations_.has_default_values())
            return arg_usage_t::input;
        if ((arg == (DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_0))
                && !attr()->scales_.get(DNNL_ARG_SRC_0).defined())
            return arg_usage_t::input;
        if ((arg == (DNNL_ARG_ATTR_INPUT_SCALES | DNNL_ARG_SRC_1))
                && !attr()->scales_.get(DNNL_ARG_SRC_1).defined())
            return arg_usage_t::input;
        if (arg == DNNL_ARG_SCRATCHPAD && !is_zero_md(scratchpad_md()))
            return arg_usage_t::output;
        for (int idx = 0; idx < attr()->post_ops_.len(); ++idx) {
            using namespace primitive_kind;
            if (post_op_has_proper_input(attr(), binary, idx, arg, DNNL_ARG_SRC_1) ||
                post_op_has_proper_input(attr(), depthwise, idx, arg, DNNL_ARG_SRC_1) ||
                post_op_has_proper_input(attr(), quantization, idx, arg, DNNL_ARG_SRC_1) ||
                post_op_has_proper_input(attr(), prelu, idx, arg, DNNL_ARG_WEIGHTS))
                return arg_usage_t::input;
        }
        if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS))
            return arg_usage_t::input;
        if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS))
            return arg_usage_t::input;

        return arg_usage_t::unused;
    }

    virtual const memory_desc_t *arg_md(int arg) const {
        // Separate binary post-ops sections due to inability to express inside
        // switch statement.
        if (arg >= DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)
                && arg < DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                           post_ops_t::post_ops_limit)) {
            const auto &po = attr()->post_ops_;
            for (int idx = 0; idx < po.len(); ++idx) {
                if (arg
                        != (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_SRC_1))
                    continue;

                return &po.entry_[idx].binary.src1_desc;
            }
        }

        switch (arg) {
            case DNNL_ARG_WORKSPACE: return workspace_md(0);
            case DNNL_ARG_SCRATCHPAD: return scratchpad_md(0);
            default: return &glob_zero_md;
        }
    }

#define DECLARE_MD_STUB(stub) \
    virtual const memory_desc_t *stub(int idx = 0) const { \
        return &glob_zero_md; \
    }

    DECLARE_MD_STUB(input_md);
    DECLARE_MD_STUB(output_md);
    DECLARE_MD_STUB(src_md);
    DECLARE_MD_STUB(diff_src_md);
    DECLARE_MD_STUB(dst_md);
    DECLARE_MD_STUB(diff_dst_md);
    DECLARE_MD_STUB(weights_md);
    DECLARE_MD_STUB(diff_weights_md);
    DECLARE_MD_STUB(workspace_md);
#undef DECLARE_MD_STUB

    const memory_desc_t *scratchpad_md(int idx = 0) const {
        return idx == 0 ? &scratchpad_md_ : &glob_zero_md;
    }

    void init_scratchpad_md() {
        auto size = scratchpad_size(scratchpad_mode::user);
        dims_t dims = {size};
        dnnl_memory_desc_init_by_tag(
                &scratchpad_md_, size ? 1 : 0, dims, data_type::u8, dnnl_x);
    }

    /** returns the scratchpad size for the given scratchpad mode. */
    dim_t scratchpad_size(scratchpad_mode_t mode) const {
        if (mode != attr_.scratchpad_mode_) return 0;
        return scratchpad_registry().size();
    }

    virtual status_t query(query_t what, int idx, void *result) const {
        auto safe_ret_md = [&](const memory_desc_t *_) {
            if (_ == nullptr) return status::not_required;
            *(const memory_desc_t **)result = _;
            return status::success;
        };

        switch (what) {
            case query::primitive_kind:
                *(primitive_kind_t *)result = kind();
                break;

            case query::memory_consumption_s64:
                *(dim_t *)result = scratchpad_size(scratchpad_mode::library);
                break;

            case query::op_d:
                if (idx != 0 || op_desc() == nullptr)
                    return status::invalid_arguments;
                *(const_c_op_desc_t *)result
                        = static_cast<const_c_op_desc_t>(op_desc());
                break;

            case query::exec_arg_md: return safe_ret_md(arg_md(idx));
            case query::src_md: return safe_ret_md(src_md(idx));
            case query::diff_src_md: return safe_ret_md(diff_src_md(idx));
            case query::dst_md: return safe_ret_md(dst_md(idx));
            case query::diff_dst_md: return safe_ret_md(diff_dst_md(idx));
            case query::weights_md: return safe_ret_md(weights_md(idx));
            case query::diff_weights_md:
                return safe_ret_md(diff_weights_md(idx));
            case query::workspace_md:
                if (idx != 0) return status::invalid_arguments;
                return safe_ret_md(workspace_md(idx));
            case query::scratchpad_md:
                if (idx != 0) return status::invalid_arguments;
                return safe_ret_md(scratchpad_md(idx));

            case query::num_of_inputs_s32: *(int *)result = n_inputs(); break;
            case query::num_of_outputs_s32: *(int *)result = n_outputs(); break;

            case query::impl_info_str: *(const char **)result = name(); break;

            default: return status::unimplemented;
        }
        return status::success;
    }

    virtual int n_inputs() const { return 0; }
    virtual int n_outputs() const { return 0; }
    int n_binary_po_inputs() const;
    int n_prelu_po_inputs() const;
    int n_depthwise_po_inputs() const;
    int n_quantization_po_inputs() const;
    // The `hint_mds(bool is_hint)` returns a vector of memory descriptors
    // that might affect the equality of primitive descriptors for backward pass.
    //
    // This function is used for creating a key to fetch primitive or primitive
    // descriptor from cache.
    //
    // 1. When creating a primitive descriptor for backward pass there may be
    //    a forward primitive descriptor hint that can be used to obtain the
    //    memory descriptors. In this case the `is_hint` argument must be `true`.
    // 2. When creating a primitive this function is called for a primitive
    //    descriptor that can be either forward or backward. In this case
    //    the `is_hint` argument must be `false`.
    //       - For forward it will return an empty vector.
    //       - For backward it will return a vector of memory descriptors if
    //         the implementation depends on a forward primitive descriptor.
    //
    // The current cases are:
    // - pooling
    // - shuffle
    //
    // Later the list of primitives can be extended. For instance, currently
    // there is no convolution on the list because nthrs + op_desc
    // (even with format=`any`) + attributes fully define a particular
    // implementation.
    virtual std::vector<memory_desc_t> hint_mds(bool is_hint) const {
        UNUSED(is_hint);
        return {};
    }

    virtual status_t create_primitive(
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
            engine_t *engine) const = 0;

    // This is a proxy interface that is used for creating nested primitives.
    // It ignores the bool value that indicates whether the requested primitive
    // was taken from cache.
    status_t create_primitive(
            std::shared_ptr<primitive_t> &primitive, engine_t *engine) const {
        std::pair<std::shared_ptr<primitive_t>, bool> p;
        CHECK(create_primitive(p, engine));
        primitive = p.first;
        return status::success;
    }

    virtual const char *name() const = 0;

    int pd_iterator_offset() const { return pd_iterator_offset_; }

protected:
    primitive_attr_t attr_;
    primitive_kind_t kind_;
    int pd_iterator_offset_;

    memory_desc_t scratchpad_md_;

    mutable pd_info_t info_;

    memory_tracking::registry_t scratchpad_registry_;

protected:
    void init_pd_iterator_offset(int offset) { pd_iterator_offset_ = offset; }

    /** compares ws between fwd_pd and this (make sense to use for bwd_pd)
     * Expectation: this already set workspace, and this workspace should
     *              exactly match the one from fwd_pd */
    bool compare_ws(const primitive_desc_t *fwd_pd) const {
        if (!workspace_md()) return true; // the impl lives fine w/o workspace
        return fwd_pd && fwd_pd->workspace_md()
                && *fwd_pd->workspace_md() == *workspace_md();
    }

    primitive_desc_t &operator=(const primitive_desc_t &other) = delete;

    /* static magic */

    template <typename pd_t>
    static status_t create(primitive_desc_t **pd, const op_desc_t *adesc,
            const primitive_attr_t *attr, engine_t *engine,
            const primitive_desc_t *hint_fwd) {
        using namespace dnnl::impl::status;
        using pd_op_desc_t = typename pkind_traits<pd_t::base_pkind>::desc_type;
        // A hack to reuse softmax code using logsoftmax primitive.
        // TODO: consider removing it in v2.0 by introducing alg_kind in softmax
        bool valid_logsoftmax = pd_t::base_pkind == primitive_kind::softmax
                && adesc->kind == primitive_kind::logsoftmax;
        bool valid_pooling = pd_t::base_pkind == primitive_kind::pooling_v2
                && adesc->kind == primitive_kind::pooling;
        if (adesc->kind != pd_t::base_pkind && !valid_logsoftmax
                && !valid_pooling)
            return invalid_arguments;
        assert(hint_fwd ? hint_fwd->kind() == pd_t::base_pkind : true);
        auto hint
                = reinterpret_cast<const typename pd_t::hint_class *>(hint_fwd);
        auto _pd = new pd_t((const pd_op_desc_t *)adesc, attr, hint);
        if (_pd == nullptr) return out_of_memory;
        if (!_pd->is_initialized()) {
            delete _pd;
            return out_of_memory;
        }
        if (_pd->init(engine) != success) {
            delete _pd;
            return unimplemented;
        }

        _pd->init_scratchpad_md();
        *pd = _pd;
        return success;
    }

    friend struct dnnl::impl::impl_list_item_t;
};

} // namespace impl
} // namespace dnnl

// dnnl_primitive_desc is a user facing entity that has an alias
// primitive_desc_iface_t for internal use.
// The primitive_desc_iface_t is responsible for holding:
// 1. impl::primitive_desc_t - a primitive descriptor implementation that
// can be stored in the primitive cache as part of the primitive implementation
// to which it belongs
// 2. engine_t - a dnnl engine
struct dnnl_primitive_desc : public dnnl::impl::c_compatible {
    dnnl_primitive_desc(const std::shared_ptr<dnnl::impl::primitive_desc_t> &pd,
            dnnl::impl::engine_t *engine);

    virtual ~dnnl_primitive_desc() = default;

    const char *info() const;
    dnnl::impl::engine_t *engine() const;
    const dnnl::impl::primitive_attr_t *attr() const;
    virtual dnnl::impl::engine_t *scratchpad_engine() const;

    virtual dnnl::impl::engine_t *src_engine() const;
    virtual dnnl::impl::engine_t *dst_engine() const;

    virtual dnnl::impl::status_t query(
            dnnl::impl::query_t what, int idx, void *result) const;

    virtual dnnl::impl::status_t create_primitive_iface(
            std::pair<primitive_iface_t *, bool> &primitive_iface) const;

    const std::shared_ptr<dnnl::impl::primitive_desc_t> &impl() const;

protected:
    std::shared_ptr<dnnl::impl::primitive_desc_t> pd_;
    dnnl::impl::engine_t *engine_;
};

#define DECLARE_COMMON_PD_t(impl_name, impl_type, use_global_scratchpad) \
    pd_t *clone() const override { \
        auto new_pd = utils::make_unique<pd_t>(*this); \
        if (!new_pd->is_initialized()) return nullptr; \
        return new_pd.release(); \
    } \
    status_t create_primitive( \
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive, \
            engine_t *engine) const override { \
        DNNL_PRIMITIVE_CREATE(pd_t) \
        return primitive_t::create_primitive_common<impl_type, pd_t>( \
                primitive, this, engine, use_global_scratchpad); \
    } \
    const char *name() const override { return impl_name; } \
    template <typename pd_t> \
    friend status_t primitive_desc_t::create(primitive_desc_t **pd, \
            const op_desc_t *adesc, const primitive_attr_t *attr, \
            engine_t *engine, const primitive_desc_t *hint_fwd);

#define DECLARE_COMMON_PD_T_USE_GLOBAL_SCRATCHPAD(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, true)

#define DECLARE_COMMON_PD_T_(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, false)

#define DECLARE_COMMON_PD_T(impl_name, impl_type, ...) \
    DECLARE_COMMON_PD_T_##__VA_ARGS__(impl_name, impl_type)

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
