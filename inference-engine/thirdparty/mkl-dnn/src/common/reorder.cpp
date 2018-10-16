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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory_pd.hpp"
#include "primitive_desc.hpp"
#include "reorder_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;

status_t mkldnn_reorder_primitive_desc_create_v2(
        primitive_desc_t **reorder_primitive_desc,
        const primitive_desc_t *input, const primitive_desc_t *output,
        const primitive_attr_t *attr) {
    bool args_ok = true
        && !any_null(reorder_primitive_desc, input, output)
        && everyone_is(primitive_kind::memory, input->kind(), output->kind());
    if (!args_ok) return invalid_arguments;

    auto i_ek = input->engine()->kind();
    auto o_ek = output->engine()->kind();
    if (!implication(i_ek != o_ek, one_of(engine_kind::cpu, i_ek, o_ek)))
        return invalid_arguments;

    auto r_pd = reinterpret_cast<reorder_pd_t **>(
            reorder_primitive_desc);
    auto i_mpd = reinterpret_cast<const memory_pd_t*>(input);
    auto o_mpd = reinterpret_cast<const memory_pd_t*>(output);

    auto i_mdw = memory_desc_wrapper(i_mpd);
    auto o_mdw = memory_desc_wrapper(o_mpd);

    if (i_mdw.nelems() == 0 || o_mdw.nelems() == 0)
        return invalid_arguments;

    if (!i_mdw.consistent_with(o_mdw))
        return invalid_arguments;

    auto e = (i_ek != engine_kind::cpu) ? input->engine() : output->engine();

    const primitive_attr_t dummy_attr;
    if (attr == NULL)
        attr = &dummy_attr;

    for (auto r = e->get_reorder_implementation_list(); *r; ++r) {
        if ((*r)(r_pd, i_mpd, o_mpd, attr) == success) {
            (*r_pd)->init_info();
            return success;
        }
    }
    return unimplemented;
}

status_t mkldnn_reorder_primitive_desc_create(
        primitive_desc_t **reorder_primitive_desc,
        const primitive_desc_t *input, const primitive_desc_t *output) {
    return mkldnn_reorder_primitive_desc_create_v2(reorder_primitive_desc,
            input, output, nullptr);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
