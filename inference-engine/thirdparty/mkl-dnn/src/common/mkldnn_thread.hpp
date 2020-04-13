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

#ifndef MKLDNN_THREAD_HPP
#define MKLDNN_THREAD_HPP

#include "utils.hpp"
#include "z_magic.hpp"

#define MKLDNN_THR_SEQ 0
#define MKLDNN_THR_OMP 1
#define MKLDNN_THR_TBB 2
#define MKLDNN_THR_TBB_AUTO 3

/* Ideally this condition below should never happen (if the library is built
 * using regular cmake). For the 3rd-party projects that build the library
 * from the sources on their own try to guess the right threading... */
#if !defined(MKLDNN_THR)
#   if defined(_OPENMP)
#       define MKLDNN_THR MKLDNN_THR_OMP
#   else
#       define MKLDNN_THR MKLDNN_THR_SEQ
#   endif
#endif

#if MKLDNN_THR == MKLDNN_THR_SEQ
#define MKLDNN_THR_SYNC 1
inline int mkldnn_get_max_threads() { return 1; }
inline int mkldnn_get_num_threads() { return 1; }
inline int mkldnn_get_thread_num() { return 0; }
inline int mkldnn_in_parallel() { return 0; }
inline void mkldnn_thr_barrier() {}

#define PRAGMA_OMP(...)

#elif MKLDNN_THR == MKLDNN_THR_OMP
#include <omp.h>
#define MKLDNN_THR_SYNC 1

inline int mkldnn_get_max_threads() { return omp_get_max_threads(); }
inline int mkldnn_get_num_threads() { return omp_get_num_threads(); }
inline int mkldnn_get_thread_num() { return omp_get_thread_num(); }
inline int mkldnn_in_parallel() { return omp_in_parallel(); }
inline void mkldnn_thr_barrier() {
#   pragma omp barrier
}

#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))

#elif (MKLDNN_THR == MKLDNN_THR_TBB || MKLDNN_THR == MKLDNN_THR_TBB_AUTO)
#include "tbb/task_arena.h"
#include "tbb/parallel_for.h"
#define MKLDNN_THR_SYNC 0

inline int mkldnn_get_max_threads()
{ return tbb::this_task_arena::max_concurrency(); }
inline int mkldnn_get_num_threads() { return mkldnn_get_max_threads(); }
inline int mkldnn_get_thread_num()
{ return tbb::this_task_arena::current_thread_index(); }
inline int mkldnn_in_parallel() { return 0; }
inline void mkldnn_thr_barrier() { assert(!"no barrier in TBB"); }

#define PRAGMA_OMP(...)

#endif

/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#   define PRAGMA_OMP_SIMD(...)
#else
#   define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__INTEL_COMPILER)

namespace mkldnn {
namespace impl {

inline bool mkldnn_thr_syncable() { return MKLDNN_THR_SYNC == 1; }

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end,
    T nx, T &nx_start, T &nx_end, T nx_divider)
{
    const int grp_count = nx_divider < nthr ? nx_divider : nthr;
    const int grp_size_big = nthr / grp_count + 1;
    const int grp_size_small = nthr / grp_count;
    const int n_grp_big = nthr % grp_count;
    const int threads_in_big_groups = n_grp_big * grp_size_big;

    const int ithr_bound_distance = ithr - threads_in_big_groups;
    T grp, grp_ithr, grp_nthr;
    if (ithr_bound_distance < 0) { // ithr in first groups
        grp = ithr / grp_size_big;
        grp_ithr = ithr % grp_size_big;
        grp_nthr = grp_size_big;
    } else { // ithr in last groups
        grp = n_grp_big + ithr_bound_distance / grp_size_small;
        grp_ithr = ithr_bound_distance % grp_size_small;
        grp_nthr = grp_size_small;
    }

    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}

} // namespace impl
} // namespace mkldnn

#include "mkldnn_thread_parallel_nd.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
