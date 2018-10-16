/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef CPU_REDUCER_HPP
#define CPU_REDUCER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_types.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "cpu_barrier.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

/** class to perform balancing over 3D array
 *
 * Conceptually the reduction happens according to the picture below:
 *
 *  <--job_size->
 *  +-----------+   +-----------+           +-----------+  ^
 *  |           |   |           |           |           |  |
 *  |           |   |           |           |           |  |
 *  |     1     |   |     2     |   . . .   |   njobs   |  | reduction_size
 *  |           |   |           |           |           |  |
 *  |           |   |           |           |           |  |
 *  +-----------+   +-----------+           +-----------+  v
 *
 *    |   |   |       |   |   |               |   |   |
 *    v   v   v       v   v   v               v   v   v
 *  ===================================================== vertical reduction
 *
 *  +-----------+   +-----------+   . . .   +-----------+ result
 *
 * In a simple case the result must be contiguous in memory.
 * @class cpu_reducer_t is an implementation.
 *
 * Threads are divided into groups. The groups are independent of each other.
 * Each group may work on several jobs (the distribution is not uniform, since
 * njobs might be not a multiple of groups). Threads within a group work on
 * different parts of the reduction dimension. Thread 0 in each group is called
 * master (@sa reduce_balancer_t::master()).
 *
 * TODO: if threading driver does not allow sync between sub-group of threads
 *       (e.g. Intel(R) TBB) enforce the # of thread per group to be 1
 */
struct reduce_balancer_t {
    reduce_balancer_t(int nthr, int job_size, int njobs, int reduction_size,
            size_t max_buffer_size)
        : syncable_(true), nthr_(nthr), job_size_(job_size), njobs_(njobs)
        , reduction_size_(reduction_size), max_buffer_size_(max_buffer_size)
    { balance(); }

    bool syncable_;
    int nthr_;
    int job_size_, njobs_, reduction_size_;

    int ngroups_; /** number of independent work (thread) groups */
    int nthr_per_group_; /** number of threads within a single work group */
    int njobs_per_group_ub_; /** the max # of jobs within a work group */

    bool master(int ithr) const { return id_in_group(ithr) == 0; }
    bool idle(int ithr) const { return ithr >= nthr_per_group_ * ngroups_; }

    int group_id(int ithr) const { return ithr / nthr_per_group_; }
    int id_in_group(int ithr) const { return ithr % nthr_per_group_; }

    int grp_njobs(int grp) const {
        if (grp >= ngroups_) return 0;
        return njobs_ / ngroups_ + (grp < njobs_ % ngroups_);
    }
    int grp_job_off(int grp) const {
        if (grp >= ngroups_) return njobs_;
        return njobs_ / ngroups_ * grp + nstl::min(grp, njobs_ % ngroups_);
    }

    int ithr_njobs(int ithr) const { return grp_njobs(group_id(ithr)); }
    int ithr_job_off(int ithr) const { return grp_job_off(group_id(ithr)); }

private:
    size_t max_buffer_size_;
    void balance();
};

/** forward declaration of reduce driver */
template <impl::data_type_t data_type> struct reducer_2d_driver_t;

/** class to perform a reduction over 3D array
 *
 * Balancing is based on @class reduce_balancer_t.
 * Restrictions: the result of the reduction must be contiguous in memory. *
 * The reduction happens according to the picture below (once more):
 *
 *  <--job_size->
 *  +-----------+   +-----------+           +-----------+  ^
 *  |           |   |           |           |           |  |
 *  |           |   |           |           |           |  |
 *  |     1     |   |     2     |   . . .   |   njobs   |  | reduction_size
 *  |           |   |           |           |           |  |
 *  |           |   |           |           |           |  |
 *  +-----------+   +-----------+           +-----------+  v
 *
 *    |   |   |       |   |   |               |   |   |
 *    v   v   v       v   v   v               v   v   v
 *  ===================================================== vertical reduction
 *
 *  +-----------+   +-----------+   . . .   +-----------+ (contiguous) result
 *
 * An example how work might be shared is shown below.
 *
 * In this example group 0 owns 2 (independent) jobs -- 2 big squares.
 * The number of threads per group is also 2 (thread 0 of group 0 and thread 1
 * of group 0). Master threads (i.e. threads with id 0 in corresponding group)
 * from each group put the partial result directly into destination memory,
 * while all the other threads with-in the group use workspace (on the picture
 * the only thread 1). Once intermediate results obtained each group reduces
 * corresponding part (own jobs) to the destination memory.
 *
 *  <-------   group 0   ------->
 *
 *  +-----------+   +-----------+  ^
 *  |           |   |           |  | thread 0 of  reduces to the dest-memory
 *  |           |   |           |  | group 0      +-----------+   +-----------+
 *  |- - - - - -|   |- - - - - -|  X
 *  |           |   |           |  | thread 1 of  reduces to workspace[tid=1]:
 *  |           |   |           |  | group 0      +-----------+   +-----------+
 *  +-----------+   +-----------+  v
 *                                                  |   |   |       |   |   |
 *                                                  v   v   v       v   v   v
 *                                   ((barrier))  =============================
 *
 *                                  dest-memory:  +-----------+   +-----------+
 */
template <impl::data_type_t data_type>
struct cpu_reducer_t {
    typedef typename prec_traits<data_type>::type data_t;

    cpu_reducer_t(const reduce_balancer_t &balancer);
    ~cpu_reducer_t();

    /** allocates internal buffer for partial computations. */
    void allocate_workspace();

    /** deallocates internal buffer. */
    void deallocate_workspace() { if (workspace_) free(workspace_); }

    /** for given thread returns the pointer where to put partial results.
     * Reduction destination @p dst must be provided as well (master threads
     * from each group will use it for partial result to reduce memory
     * pressure).
     *
     * @note: job offset is already applied by get_local_ptr(), which means all
     *        threads should start writing from the very beginning of returned
     *        address.
     */
    data_t *get_local_ptr(int ithr, data_t *dst);

    /** performs the reduction with built-in synchronization. */
    void reduce(int ithr, data_t *dst) {
        bool redundant_reduction = balancer_.nthr_per_group_ == 1
            || balancer_.idle(ithr);
        if (redundant_reduction) return;

        simple_barrier::barrier(&barriers_[balancer_.group_id(ithr)],
                balancer_.nthr_per_group_);
        reduce_nolock(ithr, dst);
    }

    reduce_balancer_t balancer_;

private:
    size_t ws_per_thread() const
    { return balancer_.njobs_per_group_ub_ * balancer_.job_size_; }

    data_t *workspace_; /** data_t[nthr_][njobs_per_group_ub_][jobs_size_] */
    reducer_2d_driver_t<data_type> *drv_;
    simple_barrier::ctx_t *barriers_; /** barrier::ctx_t[groups_] */

    void reduce_nolock(int ithr, data_t *dst);
};

template <impl::data_type_t data_type>
struct cpu_reducer_2d_t {
    typedef typename prec_traits<data_type>::type data_t;

    cpu_reducer_2d_t(const reduce_balancer_t &balancer, int job_size_x,
            int job_size_y, int x_block, int dst_x, int dst_y,
            bool master_uses_dst);
    ~cpu_reducer_2d_t();

    /** allocates internal buffer for partial computations. */
    void allocate_workspace();

    /** deallocates internal buffer. */
    void deallocate_workspace() { if (workspace_) free(workspace_); }

    /** for given thread returns the pointer where to put partial results.
     * Depending on @p master_uses_dst_ returned pointer for master threads
     * would be either equal to the destination memory or to the workspace (in
     * contrast, cpu_reducer_t struct always use destination memory for master
     * threads).
     *
     * @note: @p master_uses_dst_ == #false is unimplemented at the moment
     */
    data_t *get_local_ptr(int ithr, data_t *dst);

    /** performs the reduction with built-in synchronization. */
    void reduce(int ithr, data_t *dst) {
        bool redundant_reduction = balancer_.nthr_per_group_ == 1
            || balancer_.idle(ithr);
        if (redundant_reduction) return;

        simple_barrier::barrier(&barriers_[balancer_.group_id(ithr)],
                balancer_.nthr_per_group_);
        reduce_nolock(ithr, dst);
    }

    reduce_balancer_t balancer_;
    bool master_uses_dst_;

private:
    int job_size_x_, job_size_y_, x_block_, dst_x_, dst_y_;

    size_t ws_per_thread() const
    { return balancer_.njobs_per_group_ub_ * balancer_.job_size_; }

    data_t *workspace_; /** data_t[nthr_][njobs_per_group_ub_][jobs_size_] */
    reducer_2d_driver_t<data_type> *drv_;
    simple_barrier::ctx_t *barriers_; /** barrier::ctx_t[groups_] */

    int choose_x_blocking(int nx, int ny, int nthr_per_grp);
    void reduce_block(const data_t* wspace_base,
            data_t *dst, int job, int start_y, int start_x,
            int ny_start, int nx_start, int ny_step, int nx_step);
    void reduce_nolock(int ithr, data_t *dst);
};

/** simple 1d accumulator: y[:] += x[:] */
template <impl::data_type_t data_type>
struct cpu_accumulator_1d_t {
    typedef typename prec_traits<data_type>::type data_t;

    cpu_accumulator_1d_t();
    ~cpu_accumulator_1d_t();
    void accumulate(data_t *dst, const data_t *src, size_t size);

    reducer_2d_driver_t<data_type> *drv_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
