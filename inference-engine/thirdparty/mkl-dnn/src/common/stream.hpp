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

#ifndef STREAM_HPP
#define STREAM_HPP

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "event.hpp"
#include "engine.hpp"
#include "nstl.hpp"
#include "primitive.hpp"
#include "utils.hpp"

struct mkldnn_stream: public mkldnn::impl::c_compatible {
    typedef mkldnn::impl::nstl::vector<mkldnn::impl::primitive_t *>
        primitive_vector;

    /** a stream can have one of the following states:
     * @c running  -- state before user calls @p stream->wait(), i.e. either
     *                user is still submitting primitives or has just called @p
     *                stream->rerun(). In former case @p modifiable_ is equal
     *                to @c true, while in later case it is equal to @c false.
     * @c waiting  -- user has called stream->wait(). Computations are still in
     *                progress. @p modifiable == @c false.
     * @c stopped  -- stream finishes the execution. May happen only after
     *                calling wait(), even though the computations are done
     *                prior to that. @p modifiable == @c false.
     *
     * @todo
     *     do we really need waiting?
     *
     * @note
     *     stream state does not contain an information about error state. If
     *     some primitives are failed the stream is considered as @c
     *     submitting/rerunning before wait() is called and @c done after
     *
     * @bug
     *     for lazy stream it may happen that there is no enough memory for
     *     subsequent stream optimization -- should we consider such a stream
     *     as @c broken? This also holds even for eager stream, in case of
     *     engine->submit returns status != success... Most likely @b *NOT*. In
     *     both cases corresponding submit() or wait() will return @c
     *     status::out_of_memory leaving stream in the same state as it was
     *     before the call.
     */
    enum state_t {
        /** user is submitting a primitives or called rerun */
        running,
        /** stream->wait() was called, computations are still in progress */
        waiting,
        /** stream->wait() was called, computations are done */
        stopped,
#if 0
        /** stream is broken, no operation can be performed w/ the instance */
        broken
#endif
    };

    mkldnn_stream(mkldnn::impl::stream_kind_t kind)
        : kind_(kind), modifiable_(true), state_(mkldnn_stream::running) {}
    virtual ~mkldnn_stream() {}

    mkldnn::impl::stream_kind_t kind() const { return kind_; }

    /** submits vector of primitives @p prims to a stream
     *
     * @param prims (input)
     *   a vector of submitting primitives
     * @param error_prim (output)
     *   if not @c nullptr, address of a pointer to primitive, which will be
     *   used to indicate which primitive is caused an error, i.e.
     *   *error_primitive = &primitive_which_cause_an_error. On success the
     *   parameter is not modified.
     *
     * A high level function which is responsible for filling-in the dependency
     * graph and error checking. Implementation specific stuff happens in
     * submit_impl().
     *
     * @invariant
     *   after each submit stream is closed()
     *
     * @note
     *   To make error handling more comprehensive user must submit primitives
     *   in order of their execution (dependencies should be submitted prior to
     *   their dependents). Function closed() can be used to validate this
     *   condition
     */
    mkldnn::impl::status_t submit(const primitive_vector &prims,
            mkldnn::impl::primitive_t **error_prim);

    /** implementation specific submit for @p stream_[@p begin: @p end]
     *
     * @param begin
     *   primitive start index in stream_ (including)
     * @param end
     *   primitive end index in stream_ (not including)
     * @param error_prim (output)
     *   address of a pointer to primitive. @p error_prim != @c nullptr
     */
    virtual mkldnn::impl::status_t submit_impl(size_t begin, size_t end,
            mkldnn::impl::primitive_t **error_prim) {
        UNUSED(begin); UNUSED(end); UNUSED(error_prim);
        return mkldnn::impl::status::success;
    }

    /** returns current state */
    state_t state() const { return state_; }

    /** returns true if stream is closed, i.e. all the dependencies can be
     * resolved with-in the stream
     *
     * @todo redundant function? */
    bool closed() const;

    /** returns true if union of stream & @p prims is closed, i.e. all the
     * dependencies can be resolved with-in the stream
     *
     * @warning it is assumed closed() == @c true
     */
    bool closed(const primitive_vector &prims) const;

    /** waits until stream is done
     *
     * A high level function which is responsible for stream consistency and
     * setting state_ to @c waiting. Implementation specific stuff happens in
     * wait_impl() */
    mkldnn::impl::status_t wait(mkldnn::impl::primitive_t **error_prim);

    /** implementation specific wait */
    virtual mkldnn::impl::status_t wait_impl(
            mkldnn::impl::primitive_t **error_prim) = 0;

    /** re-runs stream
     *
     * A high level function which is responsible for stream consistency and
     * setting state_ to @c rerunning. Implementation specific stuff happens in
     * rerun_impl() */
    mkldnn::impl::status_t rerun(mkldnn::impl::primitive_t **error_prim);

    /** implementation specific rerun */
    virtual mkldnn::impl::status_t rerun_impl(
            mkldnn::impl::primitive_t **error_prim) = 0;

protected:
    mkldnn::impl::stream_kind_t kind_;
    bool modifiable_;
    state_t state_;

    primitive_vector stream_;
};

namespace mkldnn {
namespace impl {

struct stream_lazy_t;

/** \brief non-lazy stream */
struct stream_eager_t: public stream_t {
    friend stream_lazy_t;

    stream_eager_t(): stream_t(stream_kind::eager) {}

    virtual status_t submit_impl(size_t begin, size_t end,
            primitive_t **error_prim) {
        for (size_t p_index = begin; p_index < end; ++p_index) {
            primitive_t *p = stream_[p_index];
            const nstl::vector<primitive_at_t> &inputs = p->inputs();

            nstl::vector<event_t *> prereq;

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i].primitive->kind() != primitive_kind::memory) {
                    for (size_t j = 0; j < stream_.size(); ++j) {
                        if (stream_[j] == inputs[i].primitive) {
                            prereq.push_back(&deps_[inputs[i].primitive]);
                            break;
                        }
                    }
                }
            }

            status_t status = p->engine()->submit(p, &deps_[p], prereq);
            if (status != status::success) {
                *error_prim = p;
                return status;
            }
        }

        return status::success;
    }

    virtual status_t wait_impl(primitive_t **error_prim) {
        /* wait until all done */
        while (1) {
            bool execution_done = true;
            for (auto it = deps_.begin(); it != deps_.end(); ++it) {
                if (!it->second.finished()) execution_done = false;
            }
            if (execution_done) break;
        }

        /* error handling */
        for (auto it = deps_.begin(); it != deps_.end(); ++it) {
            /* XXX: topological traverse needed? */
            if (it->second.get_state() == event_t::error) {
                *error_prim = (primitive_t *)it->first;
                return status::runtime_error;
            }
        }

        return status::success;
    }

    virtual status_t rerun_impl(primitive_t **error_prim) {
        for (auto it = deps_.begin(); it != deps_.end(); ++it) {
            it->second.reset();
        }
        return submit_impl(0, stream_.size(), error_prim);
    }

protected:
    nstl::map<const primitive_t *, event_t> deps_;
};

/** \brief lazy stream
 *
 * @attention
 *     both wait_impl() and rerun_impl() may return pointer to a primitive
 *     which caused an error. Alas this @p error_prim may point to a
 *     not-user-submitted primitive because of possible fusing. It is
 *     guaranteed that the pointer will be valid till the stream is alive
 */
struct stream_lazy_t: public stream_t {
    stream_lazy_t(): stream_t(stream_kind::lazy) {}

    virtual status_t wait_impl(primitive_t **error_prim) {
        stream_eager_.stream_ = stream_;
#if 0
        for_each (aengine in stream_) {
            aengine->optimize(stream_eager_.stream_); /* in-place operation */
        }
#endif
        stream_eager_.submit(stream_eager_.stream_, error_prim);
        return stream_eager_.wait_impl(error_prim);
    }

    virtual status_t rerun_impl(primitive_t **error_prim) {
        return stream_eager_.rerun(error_prim);
    }

protected:
    stream_eager_t stream_eager_;
};

/** \brief eager_nostore stream
 *
 * This is a pseudo stream as it doesn't store the primitives inside,
 * essentially it is just a shell to run primitives by submitting them to
 * the stream
 */
struct stream_eager_nostore_t: public stream_t {
    stream_eager_nostore_t(): stream_t(stream_kind::eager_nostore) {}

    virtual status_t submit_impl(size_t begin, size_t end,
            primitive_t **error_prim) {
        UNUSED(begin);
        UNUSED(end);
        UNUSED(error_prim);
        return status::invalid_arguments;
    }

    virtual status_t wait_impl(primitive_t **error_prim) {
        UNUSED(error_prim);
        return status::success;
    }

    virtual status_t rerun_impl(primitive_t **error_prim) {
        UNUSED(error_prim);
        return status::invalid_arguments;
    }
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
