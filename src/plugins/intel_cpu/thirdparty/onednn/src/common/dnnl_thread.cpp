/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <functional>

#include "dnnl_thread.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "common/ittnotify.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "counting_barrier.hpp"
#endif

namespace dnnl {
namespace impl {

void parallel(int nthr, const std::function<void(int, int)> &f) {
    nthr = adjust_num_threads(nthr, INT64_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    assert(nthr == 1);
    f(0, 1);
#else
#if defined(DNNL_ENABLE_ITT_TASKS)
    auto task_primitive_kind = itt::primitive_task_get_current_kind();
    bool itt_enable = itt::get_itt(itt::__itt_task_level_high);
#endif
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
    {
        int nthr_ = omp_get_num_threads();
        int ithr_ = omp_get_thread_num();
        assert(nthr_ == nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
        f(ithr_, nthr_);
#if defined(DNNL_ENABLE_ITT_TASKS)
        if (ithr_ && itt_enable) itt::primitive_task_end();
#endif
    }
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    tbb::parallel_for(
            0, nthr,
            [&](int ithr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                bool mark_task = itt::primitive_task_get_current_kind()
                        == primitive_kind::undefined;
                if (mark_task && itt_enable)
                    itt::primitive_task_start(task_primitive_kind);
#endif
                f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (mark_task && itt_enable) itt::primitive_task_end();
#endif
            },
            tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB_AUTO
    tbb::parallel_for(
            0, nthr, [&](int ithr) { f(ithr, nthr); });
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    if (!tp || dnnl_in_parallel()) {
        threadpool_utils::deactivate_threadpool();
        for (int ithr = 0; ithr < nthr; ithr++) {
            f(ithr, nthr);
        }
        threadpool_utils::activate_threadpool(tp);
    } else {
        bool async = tp->get_flags()
                & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
        counting_barrier_t b;
        if (async) b.init(nthr);
        tp->parallel_for(nthr, [&, tp](int ithr, int nthr) {
            bool is_master = threadpool_utils::get_active_threadpool() == tp;
            if (!is_master) {
                threadpool_utils::activate_threadpool(tp);
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
            }
            f(ithr, nthr);
            if (!is_master) {
#if defined(DNNL_ENABLE_ITT_TASKS)
                if (itt_enable) itt::primitive_task_end();
#endif
                threadpool_utils::deactivate_threadpool();
            }
            if (async) b.notify();
        });
        if (async) b.wait();
    }
#endif
#endif
}

using F_1D_t = std::function<void(dim_t)>;
using F_2D_t = std::function<void(dim_t, dim_t)>;
using F_3D_t = std::function<void(dim_t, dim_t, dim_t)>;
using F_4D_t = std::function<void(dim_t, dim_t, dim_t, dim_t)>;
using F_5D_t = std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)>;
using F_6D_t = std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>;

using F_1D_thr_t = std::function<void(int, int, dim_t)>;
using F_2D_thr_t = std::function<void(int, int, dim_t, dim_t)>;
using F_3D_thr_t = std::function<void(int, int, dim_t, dim_t, dim_t)>;
using F_4D_thr_t = std::function<void(int, int, dim_t, dim_t, dim_t, dim_t)>;
using F_5D_thr_t
        = std::function<void(int, int, dim_t, dim_t, dim_t, dim_t, dim_t)>;
using F_6D_thr_t = std::function<void(
        int, int, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>;

void for_nd(const int ithr, const int nthr, dim_t D0, const F_1D_t &f) {
    dim_t start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (dim_t d0 = start; d0 < end; ++d0)
        f(d0);
}

void for_nd(
        const int ithr, const int nthr, dim_t D0, dim_t D1, const F_2D_t &f) {
    const dim_t work_amount = D0 * D1;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}

void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        const F_3D_t &f) {
    const dim_t work_amount = D0 * D1 * D2;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, const F_4D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4, const F_5D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4, dim_t D5, const F_6D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* for_nd_ext section */

void for_nd_ext(const int ithr, const int nthr, dim_t D0, const F_1D_thr_t &f) {
    dim_t start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (dim_t d0 = start; d0 < end; ++d0)
        f(ithr, nthr, d0);
}

void for_nd_ext(const int ithr, const int nthr, dim_t D0, dim_t D1,
        const F_2D_thr_t &f) {
    const dim_t work_amount = D0 * D1;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}

void for_nd_ext(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        const F_3D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

void for_nd_ext(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, const F_4D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

void for_nd_ext(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4, const F_5D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

void for_nd_ext(const int ithr, const int nthr, dim_t D0, dim_t D1, dim_t D2,
        dim_t D3, dim_t D4, dim_t D5, const F_6D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
    utils::nd_iterator_init(
            start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    for (dim_t iwork = start; iwork < end; ++iwork) {
        f(ithr, nthr, d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

/* parallel_nd_ext section */

void parallel_nd_ext(int nthr, dim_t D0, const F_1D_thr_t &f) {
    const dim_t work_amount = D0;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, f); });
}

void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, const F_2D_thr_t &f) {
    const dim_t work_amount = D0 * D1;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, D1, f); });
}

void parallel_nd_ext(
        int nthr, dim_t D0, dim_t D1, dim_t D2, const F_3D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, f);
        });
}

void parallel_nd_ext(
        int nthr, dim_t D0, dim_t D1, dim_t D2, dim_t D3, const F_4D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, f);
        });
}

void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
        const F_5D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}

void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
        dim_t D5, const F_6D_thr_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    nthr = adjust_num_threads(nthr, work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

/* parallel_nd section */

void parallel_nd(dim_t D0, const F_1D_t &f) {
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), D0);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
}

void parallel_nd(dim_t D0, dim_t D1, const F_2D_t &f) {
    const dim_t work_amount = D0 * D1;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, f); });
}

void parallel_nd(dim_t D0, dim_t D1, dim_t D2, const F_3D_t &f) {
    const dim_t work_amount = D0 * D1 * D2;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr,
                [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, D2, f); });
}

void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, const F_4D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, f);
        });
}

void parallel_nd(
        dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4, const F_5D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, f);
        });
}

void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4, dim_t D5,
        const F_6D_t &f) {
    const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
    int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
    if (nthr)
        parallel(nthr, [&](int ithr, int nthr) {
            for_nd(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
        });
}

} // namespace impl
} // namespace dnnl
