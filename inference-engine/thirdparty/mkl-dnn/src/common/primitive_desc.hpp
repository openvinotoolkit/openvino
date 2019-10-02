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

#ifndef PRIMITIVE_DESC_HPP
#define PRIMITIVE_DESC_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "primitive_attr.hpp"
#include "verbose.hpp"

struct mkldnn_primitive_desc: public mkldnn::impl::c_compatible {
    using memory_pd_t = mkldnn::impl::memory_pd_t;

    mkldnn_primitive_desc(mkldnn::impl::engine_t *engine,
            const mkldnn::impl::primitive_attr_t *attr,
            mkldnn::impl::primitive_kind_t kind)
        : engine_(engine), attr_(*attr), kind_(kind) { info_[0] = '\0'; }

    mkldnn_primitive_desc(mkldnn::impl::engine_t *engine,
            mkldnn::impl::primitive_kind_t kind)
        : engine_(engine), kind_(kind) { info_[0] = '\0'; }

    virtual mkldnn_primitive_desc *clone() const = 0;
    virtual ~mkldnn_primitive_desc() {}

    const mkldnn::impl::primitive_attr_t *attr() const { return &attr_; }
    mkldnn::impl::engine_t *engine() const { return engine_; }
    mkldnn::impl::primitive_kind_t kind() const { return kind_; }

    virtual void init_info() {}
    const char *info() const { return info_; }

    mkldnn::impl::memory_tracking::registry_t &scratchpad_registry()
    { return scratchpad_registry_; }
    const mkldnn::impl::memory_tracking::registry_t &scratchpad_registry() const
    { return scratchpad_registry_; }

    virtual const mkldnn::impl::op_desc_t *op_desc() const = 0;

#   define DECLARE_PD_STUB(stub) \
    virtual const memory_pd_t *stub(int idx = 0) const { return nullptr; }

    DECLARE_PD_STUB(input_pd); DECLARE_PD_STUB(output_pd);
    DECLARE_PD_STUB(src_pd); DECLARE_PD_STUB(diff_src_pd);
    DECLARE_PD_STUB(dst_pd); DECLARE_PD_STUB(diff_dst_pd);
    DECLARE_PD_STUB(weights_pd); DECLARE_PD_STUB(diff_weights_pd);
    DECLARE_PD_STUB(workspace_pd);
#   undef DECLARE_PD_STUB

    virtual int n_inputs() const { return 0; }
    virtual int n_outputs() const { return 0; }

    virtual mkldnn::impl::status_t query(mkldnn::impl::query_t what, int idx,
            void *result) const;

    virtual mkldnn::impl::status_t create_primitive(
            mkldnn::impl::primitive_t **primitive,
            const mkldnn::impl::primitive_at_t *inputs,
            const mkldnn::impl::primitive_t **outputs) const = 0;

    virtual const char *name() const { return "mkldnn_primitive_desc"; }

    /* static magic */

    template<typename pd_t>
    static mkldnn::impl::status_t create(mkldnn::impl::primitive_desc_t **pd,
            const mkldnn::impl::op_desc_t *adesc,
            const mkldnn::impl::primitive_attr_t *attr,
            mkldnn::impl::engine_t *engine,
            const mkldnn::impl::primitive_desc_t *hint_fwd) {
        using namespace mkldnn::impl;
        using namespace mkldnn::impl::status;
        using pd_op_desc_t = typename pkind_traits<pd_t::base_pkind>::desc_type;
        if (adesc->kind != pd_t::base_pkind) return invalid_arguments;
        assert(hint_fwd ? hint_fwd->kind() == pd_t::base_pkind : true);
        auto hint =
            reinterpret_cast<const typename pd_t::hint_class *>(hint_fwd);
        auto _pd = new pd_t(engine, (const pd_op_desc_t *)adesc, attr, hint);
        if (_pd == nullptr) return out_of_memory;
        if (_pd->init() != success) { delete _pd; return unimplemented; }
        _pd->init_info();
        *pd = _pd;
        return success;
    }

protected:
    mkldnn::impl::engine_t *engine_;
    mkldnn::impl::primitive_attr_t attr_;
    mkldnn::impl::primitive_kind_t kind_;

    char info_[MKLDNN_VERBOSE_BUF_LEN];

    mkldnn::impl::memory_tracking::registry_t scratchpad_registry_;
};

#define DECLARE_COMMON_PD_t(impl_name, ...) \
    virtual pd_t *clone() const override { return new pd_t(*this); } \
    virtual status_t create_primitive(primitive_t **primitive, \
            const primitive_at_t *inputs, \
            const primitive_t **outputs) const override { \
        double ms = get_msec(); \
        primitive_t::input_vector ins(inputs, inputs + this->n_inputs()); \
        primitive_t::output_vector outs(outputs, outputs + this->n_outputs()); \
        auto ret = safe_ptr_assign<primitive_t>(*primitive, \
                new (__VA_ARGS__)(this, ins, outs)); \
        ms = get_msec() - ms; \
        if (mkldnn_verbose()->level >= 2) { \
            printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms); \
            fflush(0); \
        } \
        return ret; \
    } \
    virtual const char *name() const override { return impl_name; }
#define DECLARE_COMMON_PD_T(impl_name, ...) \
    DECLARE_COMMON_PD_t(impl_name, __VA_ARGS__)

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
