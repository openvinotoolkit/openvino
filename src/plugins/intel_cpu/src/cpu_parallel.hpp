// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_threadpool.h>

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

namespace ov::intel_cpu {

// Default multiplier for the number of virtual threads when tbb partitioner is AUTO. This value is determined
// empirically.
constexpr int default_multiplier = 32;
class ThreadPool;

class CpuParallel {
public:
    CpuParallel() = delete;
    CpuParallel(CpuParallel&) = delete;
    CpuParallel(ov::intel_cpu::TbbPartitioner partitioner = ov::intel_cpu::TbbPartitioner::STATIC,
                size_t multiplier = default_multiplier);
    ~CpuParallel() = default;

    [[nodiscard]] ov::intel_cpu::TbbPartitioner get_partitioner() const {
        return m_partitioner;
    }
    [[nodiscard]] size_t get_multiplier() const {
        return m_multiplier;
    }
    [[nodiscard]] std::shared_ptr<ThreadPool> get_thread_pool() const {
        return m_thread_pool;
    }
    [[nodiscard]] int get_num_threads() const {
        int num = m_partitioner == ov::intel_cpu::TbbPartitioner::STATIC ? parallel_get_max_threads()
                                                                         : parallel_get_max_threads() * m_multiplier;
        return num;
    }
    void activate() const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        dnnl_threadpool_interop_set_max_concurrency(get_num_threads());
#endif
    }
    void parallel_simple(int D0, const std::function<void(int, int)>& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        const auto nthr = D0;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            tbb::parallel_for(0, nthr, [&](int ithr) {
                func(ithr, nthr);
            });
        } else {
            tbb::parallel_for(
                0,
                nthr,
                [&](int ithr) {
                    func(ithr, nthr);
                },
                tbb::static_partitioner());
        }
#else
        ov::parallel_for(D0, func);
#endif
    }

    template <typename T0, typename R, typename F>
    R cpu_parallel_sum(const T0& D0, const R& input, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        R res_sum = 0;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            res_sum = _TBB_REDUCE_FUNC(
                tbb::blocked_range<T0>(0, D0),
                input,
                [&](const tbb::blocked_range<T0>& r, R init) -> R {
                    R sum = init;
                    for (T0 dim1 = r.begin(); dim1 < r.end(); ++dim1) {
                        sum += func(dim1);
                    }
                    return sum;
                },
                [](R x, R y) -> R {
                    return x + y;
                });
        } else {
            res_sum = _TBB_REDUCE_FUNC(
                tbb::blocked_range<T0>(0, D0),
                input,
                [&](const tbb::blocked_range<T0>& r, R init) -> R {
                    R sum = init;
                    for (T0 dim1 = r.begin(); dim1 < r.end(); ++dim1) {
                        sum += func(dim1);
                    }
                    return sum;
                },
                [](R x, R y) -> R {
                    return x + y;
                },
                tbb::static_partitioner());
        }
        return res_sum;
#else
        return ov::parallel_sum(D0, input, func);
#endif
    }

    template <typename T0, typename F>
    void cpu_parallel_for(const T0& D0, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_1d(0, 1, D0, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_1d(ithr, virtual_threads, D0, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_1d(ithr, virtual_threads, D0, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for(D0, func);  // from core
#endif
    }

    template <typename T0, typename T1, typename F>
    void cpu_parallel_for2d(const T0& D0, const T1& D1, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0 * D1);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_2d(0, 1, D0, D1, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_2d(ithr, virtual_threads, D0, D1, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_2d(ithr, virtual_threads, D0, D1, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for2d(D0, D1, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename F>
    void cpu_parallel_for3d(const T0& D0, const T1& D1, const T2& D2, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0 * D1 * D2);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_3d(0, 1, D0, D1, D2, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_3d(ithr, virtual_threads, D0, D1, D2, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_3d(ithr, virtual_threads, D0, D1, D2, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for3d(D0, D1, D2, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename F>
    void cpu_parallel_for4d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0 * D1 * D2 * D3);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_4d(0, 1, D0, D1, D2, D3, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_4d(ithr, virtual_threads, D0, D1, D2, D3, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_4d(ithr, virtual_threads, D0, D1, D2, D3, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for4d(D0, D1, D2, D3, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
    void cpu_parallel_for5d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const T4& D4, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0 * D1 * D2 * D3 * D4);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_5d(0, 1, D0, D1, D2, D3, D4, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_5d(ithr, virtual_threads, D0, D1, D2, D3, D4, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_5d(ithr, virtual_threads, D0, D1, D2, D3, D4, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for5d(D0, D1, D2, D3, D4, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
    void cpu_parallel_for6d(const T0& D0,
                            const T1& D1,
                            const T2& D2,
                            const T3& D3,
                            const T4& D4,
                            const T5& D5,
                            const F& func) const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        auto work_amount = static_cast<int>(D0 * D1 * D2 * D3 * D4 * D5);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * m_multiplier;
        }
        if (virtual_threads > work_amount) {
            virtual_threads = work_amount;
        }
        if (virtual_threads == 1) {
            for_6d(0, 1, D0, D1, D2, D3, D4, D5, func);
        } else {
            if (m_partitioner == ov::intel_cpu::TbbPartitioner::AUTO) {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_6d(ithr, virtual_threads, D0, D1, D2, D3, D4, D5, func);
                });
            } else {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_6d(ithr, virtual_threads, D0, D1, D2, D3, D4, D5, func);
                    },
                    tbb::static_partitioner());
            }
        }
#else
        ov::parallel_for6d(D0, D1, D2, D3, D4, D5, func);
#endif
    }

    template <typename T0, typename R, typename F>
    R parallel_sum(const T0& D0, const R& input, const F& func) const {
        return cpu_parallel_sum(D0, input, func);
    }
    template <typename T0, typename F>
    void parallel_for(const T0& D0, const F& func) const {
        cpu_parallel_for(D0, func);
    }
    template <typename T0, typename T1, typename F>
    void parallel_for2d(const T0& D0, const T1& D1, const F& func) const {
        cpu_parallel_for2d(D0, D1, func);
    }
    template <typename T0, typename T1, typename T2, typename F>
    void parallel_for3d(const T0& D0, const T1& D1, const T2& D2, const F& func) const {
        cpu_parallel_for3d(D0, D1, D2, func);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename F>
    void parallel_for4d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const F& func) const {
        cpu_parallel_for4d(D0, D1, D2, D3, func);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
    void parallel_for5d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const T4& D4, const F& func) const {
        cpu_parallel_for5d(D0, D1, D2, D3, D4, func);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
    void parallel_for6d(const T0& D0,
                        const T1& D1,
                        const T2& D2,
                        const T3& D3,
                        const T4& D4,
                        const T5& D5,
                        const F& func) const {
        cpu_parallel_for6d(D0, D1, D2, D3, D4, D5, func);
    }

private:
    ov::intel_cpu::TbbPartitioner m_partitioner = ov::intel_cpu::TbbPartitioner::STATIC;
    size_t m_multiplier = default_multiplier;
    std::shared_ptr<ThreadPool> m_thread_pool = nullptr;
};

}  // namespace ov::intel_cpu
