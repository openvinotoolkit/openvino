// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

enum class Partitioner {
    STATIC,
    AUTO,
};

class CpuParallel {
public:
    CpuParallel() = default;
    CpuParallel(Partitioner partitioner, size_t multiplier)
        : m_default_partitioner(partitioner),
          m_default_multiplier(multiplier) {}

    Partitioner get_partitioner() const {
        return m_default_partitioner;
    }
    size_t get_multiplier() {
        return m_default_multiplier;
    }

    template <typename T0, typename F>
    void parallel_simple(const T0& D0, const F& func) const {
#if OV_THREAD == OV_THREAD_TBB
        const int nthr = static_cast<size_t>(D0);
        if (m_default_partitioner == Partitioner::STATIC) {
            tbb::parallel_for(
                0,
                nthr,
                [&](int ithr) {
                    func(ithr, nthr);
                },
                tbb::static_partitioner());
        } else {
            tbb::parallel_for(0, nthr, [&](int ithr) {
                func(ithr, nthr);
            });
        }
#else
        ov::parallel_for(D0, func);  // from core
#endif
    }

    template <typename T0, typename F>
    void parallel_for(const T0& D0, const F& func, const Partitioner& partitioner, const size_t multiplier)
        const {  // may be a variadical template to handle multidimmentional use cases
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        }
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_1d(0, 1, D0, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_1d(ithr, virtual_threads, D0, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_1d(ithr, virtual_threads, D0, func);
                });
            }
        }
#else
        ov::parallel_for(D0, func);  // from core
#endif
    }

    template <typename T0, typename T1, typename F>
    void parallel_for2d(const T0& D0,
                        const T1& D1,
                        const F& func,
                        const Partitioner& partitioner,
                        const size_t multiplier) {
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0 * D1);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        }
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_2d(0, 1, D0, D1, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_2d(ithr, virtual_threads, D0, D1, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_2d(ithr, virtual_threads, D0, D1, func);
                });
            }
        }
#else
        ov::parallel_for2d(D0, D1, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename F>
    void parallel_for3d(const T0& D0,
                        const T1& D1,
                        const T2& D2,
                        const F& func,
                        const Partitioner& partitioner,
                        const size_t multiplier) {
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0 * D1 * D2);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO) {
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        }
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_3d(0, 1, D0, D1, D2, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_3d(ithr, virtual_threads, D0, D1, D2, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_3d(ithr, virtual_threads, D0, D1, D2, func);
                });
            }
        }
#else
        ov::parallel_for3d(D0, D1, D2, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename F>
    void parallel_for4d(const T0& D0,
                        const T1& D1,
                        const T2& D2,
                        const T3& D3,
                        const F& func,
                        const Partitioner& partitioner,
                        const size_t multiplier) {
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0 * D1 * D2 * D3);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO)
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_4d(0, 1, D0, D1, D2, D3, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_4d(ithr, virtual_threads, D0, D1, D2, D3, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_4d(ithr, virtual_threads, D0, D1, D2, D3, func);
                });
            }
        }
#else
        ov::parallel_for4d(D0, D1, D2, D3, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
    void parallel_for5d(const T0& D0,
                        const T1& D1,
                        const T2& D2,
                        const T3& D3,
                        const T4& D4,
                        const F& func,
                        const Partitioner& partitioner,
                        const size_t multiplier) {
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0 * D1 * D2 * D3 * D4);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO)
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_5d(0, 1, D0, D1, D2, D3, D4, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_5d(ithr, virtual_threads, D0, D1, D2, D3, D4, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_5d(ithr, virtual_threads, D0, D1, D2, D3, D4, func);
                });
            }
        }
#else
        ov::parallel_for5d(D0, D1, D2, D3, D4, func);
#endif
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
    void parallel_for6d(const T0& D0,
                        const T1& D1,
                        const T2& D2,
                        const T3& D3,
                        const T4& D4,
                        const T5& D5,
                        const F& func,
                        const Partitioner& partitioner,
                        const size_t multiplier) {
#if OV_THREAD == OV_THREAD_TBB
        auto work_amount = static_cast<size_t>(D0 * D1 * D2 * D3 * D4 * D5);
        const int nthr = parallel_get_max_threads();
        int virtual_threads = nthr;
        if (partitioner == Partitioner::AUTO)
            virtual_threads = 1 == nthr ? 1 : nthr * multiplier;
        if (static_cast<size_t>(virtual_threads) > work_amount)
            virtual_threads = static_cast<int>(work_amount);
        if (virtual_threads == 1) {
            for_6d(0, 1, D0, D1, D2, D3, D4, D5, func);
        } else {
            if (partitioner == Partitioner::STATIC) {
                tbb::parallel_for(
                    0,
                    virtual_threads,
                    [&](int ithr) {
                        for_6d(ithr, virtual_threads, D0, D1, D2, D3, D4, D5, func);
                    },
                    tbb::static_partitioner());
            } else {
                tbb::parallel_for(0, virtual_threads, [&](int ithr) {
                    for_6d(ithr, virtual_threads, D0, D1, D2, D3, D4, D5, func);
                });
            }
        }
#else
        ov::parallel_for6d(D0, D1, D2, D3, D4, D5, func);
#endif
    }

    template <typename T0, typename F>
    void parallel_for(const T0& D0, const F& func) const {
        parallel_for(D0, func, m_default_partitioner, m_default_multiplier);
    }
    template <typename T0, typename T1, typename F>
    void parallel_for2d(const T0& D0, const T1& D1, const F& func) {
        parallel_for2d(D0, D1, func, m_default_partitioner, m_default_multiplier);
    }
    template <typename T0, typename T1, typename T2, typename F>
    void parallel_for3d(const T0& D0, const T1& D1, const T2& D2, const F& func) {
        parallel_for3d(D0, D1, D2, func, m_default_partitioner, m_default_multiplier);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename F>
    void parallel_for4d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const F& func) {
        parallel_for4d(D0, D1, D2, D3, func, m_default_partitioner, m_default_multiplier);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
    void parallel_for5d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const T4& D4, const F& func) {
        parallel_for5d(D0, D1, D2, D3, D4, func, m_default_partitioner, m_default_multiplier);
    }
    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
    void
    parallel_for6d(const T0& D0, const T1& D1, const T2& D2, const T3& D3, const T4& D4, const T5& D5, const F& func) {
        parallel_for6d(D0, D1, D2, D3, D4, D5, func, m_default_partitioner, m_default_multiplier);
    }

private:
    Partitioner m_default_partitioner = Partitioner::STATIC;
    size_t m_default_multiplier = 32;
};

}  // namespace ov::intel_cpu
