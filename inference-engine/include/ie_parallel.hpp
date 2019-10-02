// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and definitions for sequential and multi-threading implementations.
 * Multi-threading support is implemented in two variants: using the Threading Building Blocks library and OpenMP* product.
 * To build a particular implementation, use the corresponding identifier: IE_THREAD_TBB, IE_THREAD_TBB_AUTO, IE_THREAD_OMP or IE_THREAD_SEQ.
 * @file ie_parallel.hpp
 */

#pragma once

#include <cstddef>

#define IE_THREAD_TBB 0
#define IE_THREAD_OMP 1
#define IE_THREAD_SEQ 2
#define IE_THREAD_TBB_AUTO 3

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include "tbb/task_scheduler_observer.h"
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include "tbb/parallel_sort.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"

inline int  parallel_get_max_threads() { return tbb::this_task_arena::max_concurrency(); }
inline int  parallel_get_num_threads() { return parallel_get_max_threads(); }
inline int  parallel_get_thread_num()  { return tbb::this_task_arena::current_thread_index(); }
inline void parallel_set_num_threads(int n) { return; }
inline int  parallel_get_env_threads() { return 0; }
#if IE_THREAD == IE_THREAD_TBB
    #define PARTITIONING , tbb::static_partitioner()
#else
    #define PARTITIONING
#endif
#elif IE_THREAD == IE_THREAD_OMP
#include <algorithm>
#include <cstdlib>
#include <string>
#include <omp.h>


/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#endif  // defined(_MSC_VER) && !defined(__INTEL_COMPILER)
inline int  parallel_get_max_threads() { return omp_get_max_threads(); }
inline int  parallel_get_num_threads() { return omp_get_num_threads(); }
inline int  parallel_get_thread_num()  { return omp_get_thread_num(); }
inline void parallel_set_num_threads(int n) { omp_set_num_threads(n); }
inline int  parallel_get_env_threads() {
    int env_cores = 0;
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        try {
            env_cores = std::stoi(getenv("OMP_NUM_THREADS"));
        } catch (const std::exception&) {
            env_cores = 0;
        }
    }
    return env_cores;
}

#elif IE_THREAD == IE_THREAD_SEQ
#include <algorithm>  // NOLINT
inline int  parallel_get_env_threads() { return 1; }
inline int  parallel_get_max_threads() { return 1; }
inline int  parallel_get_num_threads() { return 1; }
inline int  parallel_get_thread_num()  { return 0; }
inline void parallel_set_num_threads(int n) { return; }
#endif


namespace InferenceEngine {

template <typename F>
void parallel_nt(int nthr, const F &func) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    if (nthr == 0) nthr = parallel_get_max_threads();
    if (nthr == 1) {
        func(0, 1);
        return;
    }

    tbb::parallel_for(0, nthr, [&](int ithr) {
        func(ithr, nthr);
    });
#elif IE_THREAD == IE_THREAD_OMP
    if (nthr == 1) {
        func(0, 1);
        return;
    }

#   pragma omp parallel num_threads(nthr)
    func(parallel_get_thread_num(), parallel_get_num_threads());
#elif IE_THREAD == IE_THREAD_SEQ
    func(0, 1);
#endif
}

template <typename F>
void parallel_nt_static(int nthr, const F &func) {
#if IE_THREAD == IE_THREAD_SEQ
    const bool serial = true;
#else
    const bool serial = false;
#endif

    if (serial || nthr == 1) {
        func(0, 1);
        return;
    }

    if (nthr == 0) nthr = parallel_get_max_threads();
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    tbb::parallel_for(0, nthr, [&](int ithr) {
            func(ithr, nthr);
        }
        , tbb::static_partitioner{});

#elif IE_THREAD == IE_THREAD_OMP

#   pragma omp parallel num_threads(nthr)
    {
        func(parallel_get_thread_num(), parallel_get_num_threads());
    }
#endif
}

template <typename I, typename F>
void parallel_sort(I begin, I end, const F &comparator) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    tbb::parallel_sort(begin, end, comparator);
#elif IE_THREAD == IE_THREAD_OMP
    // TODO: propose OpenMP version
    std::sort(begin, end, comparator);
#elif IE_THREAD == IE_THREAD_SEQ
    std::sort(begin, end, comparator);
#endif
}

template <typename T0, typename R, typename F>
R parallel_sum(const T0 &D0, const R &input, const F &func) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    return tbb::parallel_reduce(
        tbb::blocked_range<T0>(0, D0), input,
        [&](const tbb::blocked_range<T0>& r, R init)->R {
            R sum = init;
            for (T0 dim1 = r.begin(); dim1 < r.end(); ++dim1)
                sum += func(dim1);
            return sum;
        },
        [](R x, R y)->R {
            return x + y;
        } PARTITIONING);
#else
    R sum = input;

#ifdef _MSC_VER
    using T0_IT = typename std::make_signed<T0>::type;
#else
    using T0_IT = T0;
#endif

#if IE_THREAD == IE_THREAD_OMP
    #pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
    for (T0_IT dim1 = 0; dim1 < static_cast<T0_IT>(D0); dim1++) {
        sum += static_cast<R>(func(dim1));
    }
    return sum;
#endif
}

template <typename T0, typename T1, typename R, typename F>
R parallel_sum2d(const T0 &D0, const T1 &D1, const R &input, const F &func) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    return tbb::parallel_reduce(
        tbb::blocked_range2d<T0, T1>(0, D0, 0, D1), input,
        [&](const tbb::blocked_range2d<T0, T1>& r, R init)->R {
            R sum = init;
            for (T0 dim2 = r.rows().begin(); dim2 < r.rows().end(); dim2++) {
                for (T1 dim1 = r.cols().begin(); dim1 < r.cols().end(); dim1++) {
                    sum += func(dim2, dim1);
                }
            }
            return sum;
        },
        [](R x, R y)->R {
            return x + y;
        } PARTITIONING);
#else
    R sum = input;

#ifdef _MSC_VER
    using T0_IT = typename std::make_signed<T0>::type;
    using T1_IT = typename std::make_signed<T1>::type;
#else
    using T0_IT = T0;
    using T1_IT = T1;
#endif

#if IE_THREAD == IE_THREAD_OMP
    #pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
#endif
    for (T0_IT dim2 = 0; dim2 < D0; dim2++) {
        for (T1_IT dim1 = 0; dim1 < D1; dim1++) {
            sum += func(dim2, dim1);
        }
    }
    return sum;
#endif
}
template <typename T0, typename T1, typename T2, typename R, typename F>
R parallel_sum3d(const T0 &D0, const T1 &D1, const T2 &D2, const R &input, const F &func) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    return tbb::parallel_reduce(
        tbb::blocked_range3d<T0, T1, T2>(0, D0, 0, D1, 0, D2), input,
        [&](const tbb::blocked_range3d<T0, T1, T2>& r, R init)->R {
            R sum = init;
            for (T0 dim1 = r.pages().begin(); dim1 < r.pages().end(); dim1++) {
                for (T1 dim2 = r.rows().begin(); dim2 < r.rows().end(); dim2++) {
                    for (T2 dim3 = r.cols().begin(); dim3 < r.cols().end(); dim3++) {
                        sum += func(dim1, dim2, dim3);
                    }
                }
            }
            return sum;
        },
        [](R x, R y)->R {
            return x + y;
        } PARTITIONING);
#else
    R sum = input;

#ifdef _MSC_VER
    using T0_IT = typename std::make_signed<T0>::type;
    using T1_IT = typename std::make_signed<T1>::type;
    using T2_IT = typename std::make_signed<T2>::type;
#else
    using T0_IT = T0;
    using T1_IT = T1;
    using T2_IT = T2;
#endif

#if IE_THREAD == IE_THREAD_OMP
    #pragma omp parallel for collapse(3) reduction(+ : sum) schedule(static)
#endif
    for (T0_IT dim1 = 0; dim1 < static_cast<T0_IT>(D0); dim1++) {
        for (T1_IT dim2 = 0; dim2 < static_cast<T1_IT>(D1); dim2++) {
            for (T2_IT dim3 = 0; dim3 < static_cast<T2_IT>(D2); dim3++) {
                sum += func(dim1, dim2, dim3);
            }
        }
    }
    return sum;
#endif
}

template<typename T>
inline T parallel_it_init(T start) { return start; }
template<typename T, typename Q, typename R, typename... Args>
inline T parallel_it_init(T start, Q &x, const R &X, Args &&... tuple) {
    start = parallel_it_init(start, static_cast<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool parallel_it_step() { return true; }
template<typename Q, typename R, typename... Args>
inline bool parallel_it_step(Q &x, const R &X, Args &&... tuple) {
    if (parallel_it_step(static_cast<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template <typename T, typename Q>
inline void splitter(const T &n, const Q &team, const Q &tid, T &n_start, T &n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}


template <typename T0, typename F>
void for_1d(const int &ithr, const int &nthr, const T0 &D0, const F &func) {
    T0 d0{ 0 }, end{ 0 };
    splitter(D0, nthr, ithr, d0, end);
    for (; d0 < end; ++d0) func(d0);
}

template <typename T0, typename F>
void parallel_for(const T0 &D0, const F &func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_1d(0, 1, D0, func);
    } else {
        tbb::parallel_for(0, nthr, [&](int ithr) {
            for_1d(ithr, nthr, D0, func);
        }, tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_1d(ithr, nthr, D0, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_1d(parallel_get_thread_num(), parallel_get_num_threads(), D0, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_1d(0, 1, D0, func);
#endif
}


template <typename T0, typename T1, typename F>
void for_2d(const int &ithr, const int &nthr, const T0 &D0, const T1 &D1, const F &func) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 };
    parallel_it_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1);
        parallel_it_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename F>
void parallel_for2d(const T0 &D0, const T1 &D1, const F &func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0 * D1);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_2d(0, 1, D0, D1, func);
    } else {
        tbb::parallel_for(0, nthr, [&](int ithr) {
            for_2d(ithr, nthr, D0, D1, func);
        }, tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_2d(ithr, nthr, D0, D1, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_2d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_2d(0, 1, D0, D1, func);
#endif
}


template <typename T0, typename T1, typename T2, typename F>
void for_3d(const int &ithr, const int &nthr, const T0 &D0, const T1 &D1,
    const T2 &D2, const F &func) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2);
        parallel_it_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_for3d(const T0 &D0, const T1 &D1, const T2 &D2, const F &func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0 * D1 * D2);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_3d(0, 1, D0, D1, D2, func);
    } else {
        tbb::parallel_for(0, nthr, [&](int ithr) {
            for_3d(ithr, nthr, D0, D1, D2, func);
        }, tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_3d(ithr, nthr, D0, D1, D2, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_3d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_3d(0, 1, D0, D1, D2, func);
#endif
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_4d(const int &ithr, const int &nthr, const T0 &D0, const T1 &D1,
    const T2 &D2, const T3 &D3, const F &func) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 }; T3 d3{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2, d3);
        parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_for4d(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, const F &func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0 * D1 * D2 * D3);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_4d(0, 1, D0, D1, D2, D3, func);
    } else {
        tbb::parallel_for(0, nthr, [&](int ithr) {
            for_4d(ithr, nthr, D0, D1, D2, D3, func);
        }, tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_4d(ithr, nthr, D0, D1, D2, D3, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_4d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, D3, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_4d(0, 1, D0, D1, D2, D3, func);
#endif
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
void for_5d(const int &ithr, const int &nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const F &func) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 }; T3 d3{ 0 }; T4 d4{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2, d3, d4);
        parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
void parallel_for5d(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
                    const T4 &D4, const F &func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0 * D1 * D2 * D3 * D4);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_5d(0, 1, D0, D1, D2, D3, D4, func);
    } else {
        tbb::parallel_for(0, nthr, [&](int ithr) {
            for_5d(ithr, nthr, D0, D1, D2, D3, D4, func);
        }, tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_5d(ithr, nthr, D0, D1, D2, D3, D4, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_5d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, D3, D4, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_5d(0, 1, D0, D1, D2, D3, D4, func);
#endif
}

}  // namespace InferenceEngine

