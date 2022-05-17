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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "primitive_cache.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "reorder_pd.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {

namespace {
engine_t *get_reorder_engine(engine_t *src_engine, engine_t *dst_engine) {
    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    auto s_rk = src_engine->runtime_kind();
    auto d_rk = dst_engine->runtime_kind();

    if (is_native_runtime(d_rk)) return src_engine;

    if (is_native_runtime(s_rk)) return dst_engine;

    if (d_ek == engine_kind::cpu) return src_engine;

    if (s_ek == engine_kind::cpu) return dst_engine;

    assert(s_ek == engine_kind::gpu);
    assert(d_ek == engine_kind::gpu);
    return src_engine;
}
} // namespace

status_t reorder_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        engine_t *engine, const memory_desc_t *src_md, engine_t *src_engine,
        const memory_desc_t *dst_md, engine_t *dst_engine,
        const primitive_attr_t *attr) {
    pd.reset();

    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    if (!IMPLICATION(s_ek != d_ek, utils::one_of(engine_kind::cpu, s_ek, d_ek)))
        return invalid_arguments;

    auto s_mdw = memory_desc_wrapper(*src_md);
    auto d_mdw = memory_desc_wrapper(*dst_md);

    if (!s_mdw.consistent_with(d_mdw))
        return invalid_arguments;

    if (attr == nullptr) attr = &default_attr();

    bool is_cross_engine = src_engine != dst_engine
            && utils::one_of(
                    engine_kind::gpu, src_engine->kind(), dst_engine->kind());

    dnnl_reorder_desc_t desc = {primitive_kind::reorder, src_md, dst_md, s_ek,
            d_ek, is_cross_engine};
    primitive_hashing::key_t key(
            engine, reinterpret_cast<op_desc_t *>(&desc), attr, 0, {});
    pd = primitive_cache().get_pd(key);
    if (pd) return success;

    for (auto r = engine->get_reorder_implementation_list(src_md, dst_md); *r;
            ++r) {
        reorder_pd_t *reorder_pd = nullptr;
        if ((*r)(&reorder_pd, engine, attr, src_engine, src_md, dst_engine,
                    dst_md)
                == success) {
            pd.reset(reorder_pd);
            return success;
        }
    }
    return unimplemented;
}

status_t reorder_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr) {
    return reorder_primitive_desc_create(
            pd, engine, src_md, engine, dst_md, engine, attr);
}

} // namespace impl
} // namespace dnnl

status_t dnnl_reorder_primitive_desc_create(
        primitive_desc_iface_t **reorder_pd_iface, const memory_desc_t *src_md,
        engine_t *src_engine, const memory_desc_t *dst_md, engine_t *dst_engine,
        const primitive_attr_t *attr) {
    if (any_null(reorder_pd_iface, src_engine, src_md, dst_engine, dst_md))
        return invalid_arguments;

    std::shared_ptr<primitive_desc_t> pd;
    auto e = get_reorder_engine(src_engine, dst_engine);
    CHECK(reorder_primitive_desc_create(
            pd, e, src_md, src_engine, dst_md, dst_engine, attr));

    return safe_ptr_assign(*reorder_pd_iface,
            new reorder_primitive_desc_iface_t(pd, e, src_engine, dst_engine));
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
