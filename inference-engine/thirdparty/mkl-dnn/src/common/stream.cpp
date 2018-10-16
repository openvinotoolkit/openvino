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
#include "nstl.hpp"
#include "stream.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

status_t stream_t::submit(const nstl::vector<primitive_t *> &prims,
        primitive_t **error_prim) {
    if (!modifiable_) return invalid_arguments;

    primitive_t *error_primitive_stub;
    if (error_prim == nullptr) error_prim = &error_primitive_stub;

    /* check whether adding each new primitive stream is always closed */
    nstl::vector<primitive_t *> tmp;
    for (size_t i = 0; i < prims.size(); ++i) {
        tmp.push_back(prims[i]);
        if (!closed(tmp)) {
            *error_prim = prims[i];
            return invalid_arguments;
        }
    }

    const size_t start = stream_.size();
    stream_.insert(stream_.end(), prims.begin(), prims.end());
    return submit_impl(start, stream_.size(), error_prim);
}

bool stream_t::closed() const { return true; }

bool stream_t::closed(const primitive_vector &prims) const { return true; }

status_t stream_t::wait(primitive_t **error_prim) {
    if (!closed()) return invalid_arguments; /* XXX: redundant? */

    primitive_t *error_primitive_stub;
    if (error_prim == nullptr) error_prim = &error_primitive_stub;

    modifiable_ = false;
    state_ = stream_t::waiting;
    status_t status = wait_impl(error_prim);
    state_ = stream_t::stopped;
    return status;
}

status_t stream_t::rerun(primitive_t **error_prim) {
    if (state() != stream_t::stopped) return invalid_arguments;

    primitive_t *error_primitive_stub;
    if (error_prim == nullptr) error_prim = &error_primitive_stub;

    state_ = stream_t::running;
    return rerun_impl(error_prim);
}

/* API */

status_t mkldnn_stream_create(stream_t **stream, stream_kind_t stream_kind) {
    bool args_ok = stream != nullptr && utils::one_of(stream_kind,
            stream_kind::eager, stream_kind::lazy);
    if (!args_ok)
        return invalid_arguments;

    stream_t *s;
    if (stream_kind == stream_kind::eager)
        s = new stream_eager_t;
    else
        s = new stream_lazy_t;
    return safe_ptr_assign<stream_t>(*stream, s);
}

status_t mkldnn_stream_submit(stream_t *stream, size_t n,
        primitive_t *primitives[], primitive_t **error_primitive) {
    bool args_ok = !utils::any_null(stream, primitives);
    if (!args_ok) return invalid_arguments;

    nstl::vector<primitive_t *> prims;
    for (size_t i = 0; i < n; ++i) {
        if (primitives[i] == nullptr) return invalid_arguments;
        prims.push_back(primitives[i]);
    }
    return stream->submit(prims, error_primitive);
}

status_t mkldnn_stream_wait(stream_t *stream, int block,
        primitive_t **error_primitive) {
    UNUSED(block);
    if (stream == nullptr) return invalid_arguments;
    return stream->wait(error_primitive);
}

status_t mkldnn_stream_rerun(stream_t *stream, primitive_t **error_primitive) {
    if (stream == nullptr) return invalid_arguments;
    return stream->rerun(error_primitive);
}

status_t mkldnn_stream_destroy(stream_t *stream) {
    if (stream) delete stream;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
