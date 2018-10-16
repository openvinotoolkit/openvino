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

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "event.hpp"
#include "primitive.hpp"
#include "utils.hpp"

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Run a set of primitives according to an execution order (a partially
 *      ordered set represented by events)
 *   - Provide engine specific primitive_desc_t creators
 *
 * Implementation specifics:
 *   - Engine doesn't own any of primitives or events -- it is up to caller to
 *      guarantee all the pointers are valid until execution is finished
 */
struct mkldnn_engine: public mkldnn::impl::c_compatible {
    mkldnn_engine(mkldnn::impl::engine_kind_t kind)
        : kind_(kind)
    {}
    virtual ~mkldnn_engine() {}

    typedef mkldnn::impl::nstl::vector<mkldnn::impl::event_t *>
        event_vector;

#if 0
    /** reduce ref counting for current engine */
    virtual void dec_ref_count() = 0;
#endif

    /** get kind of the current engine */
    virtual mkldnn::impl::engine_kind_t kind() const { return kind_; }

    /** submits a primitive @p p for execution
     *
     * @param p (input)
     *   primitive to execute
     * @param e (output)
     *   resulting event (to be passed to p->execute(e))
     * @param prerequisites (input)
     *   vector of prerequisite events that must be finished before @p p is run
     *
     * @return
     *   status of the operation
     *
     * @remark @b Rational.
     *   Prerequisites are separated from input-resources. Though memory is a
     *   primitive, it becomes a singularity point in the sense of signaling
     *   that it is ready (either it should not have corresponding event or the
     *   event should be always returns it is ready). Let engine has pretty
     *   simple logic wrt primitive run. Also this approach allows to reduce
     *   the amount of prerequisites checks -- usually the \# of real
     *   dependencies < the \# of primitive inputs).
     *
     * @warning
     *   Engine does not track dependencies and their consistencies. Internal
     *   library code may easily submit a primitive with the same resulting and
     *   prerequisite event, obtaining dead-lock. Engine won't even try to
     *   prevent such a situation.
     *
     * @note
     *   if any of @p prerequisites is finished with @c event::error or @c
     *   event::aborted primitive @p p would not be executed, its event's @p e
     *   state is automatically set to @c event::aborted */
    virtual mkldnn::impl::status_t submit(mkldnn::impl::primitive_t *p,
            mkldnn::impl::event_t *e, event_vector &prerequisites) = 0;

    /* implementation section */
    virtual mkldnn::impl::status_t memory_primitive_desc_create(
            mkldnn::impl::memory_pd_t **memory_pd,
            const mkldnn::impl::memory_desc_t *memory_d)
    { return mkldnn::impl::status::unimplemented; }

    virtual mkldnn::impl::status_t view_primitive_desc_create(
            mkldnn::impl::view_pd_t **view_pd,
            const mkldnn::impl::memory_pd_t *memory_pd,
            const mkldnn::impl::dims_t dims,
            const mkldnn::impl::dims_t offsets)
    { return mkldnn::impl::status::unimplemented; }

    typedef mkldnn::impl::status_t (*concat_primitive_desc_create_f)(
            mkldnn::impl::concat_pd_t **concat_pd,
            const mkldnn::impl::memory_desc_t *output_d, int n, int concat_dim,
            const mkldnn::impl::memory_pd_t **input_pds,
            const mkldnn::impl::primitive_attr_t *attr);

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const concat_primitive_desc_create_f*
        get_concat_implementation_list() const;

    typedef mkldnn::impl::status_t (*sum_primitive_desc_create_f)(
            mkldnn::impl::sum_pd_t **sum_pd,
            const mkldnn::impl::memory_desc_t *output_d, int n,
            const float *scales, const mkldnn::impl::memory_pd_t **input_pds,
            const mkldnn::impl::primitive_attr_t *attr);

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const sum_primitive_desc_create_f*
        get_sum_implementation_list() const;

    typedef mkldnn::impl::status_t (*reorder_primitive_desc_create_f)(
            mkldnn::impl::reorder_pd_t **reorder_pd,
            const mkldnn::impl::memory_pd_t *input_memory_pd,
            const mkldnn::impl::memory_pd_t *output_memory_pd,
            const mkldnn::impl::primitive_attr_t *attr);

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const reorder_primitive_desc_create_f*
        get_reorder_implementation_list() const;

    typedef mkldnn::impl::status_t (*primitive_desc_create_f)(
            mkldnn::impl::primitive_desc_t **, const mkldnn::impl::op_desc_t *,
            const mkldnn::impl::primitive_attr_t *attr,
            mkldnn::impl::engine_t *, const mkldnn::impl::primitive_desc_t *);

    /** return the list of implementations. engine guarantees to return a
     * NULL-terminated list */
    virtual const primitive_desc_create_f* get_implementation_list() const;

protected:
    mkldnn::impl::engine_kind_t kind_;
};

namespace mkldnn {
namespace impl {

struct engine_factory_t: public c_compatible {
    virtual size_t count() const = 0;
    virtual engine_kind_t kind() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
