// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <functional>

namespace ov {
namespace cpu {

size_t getTotalThreads();
void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);

// copy from openvino/core/parallel.hpp
template <typename T>
inline T parallel_it_init(T start) {
    return start;
}
template <typename T, typename Q, typename R, typename... Args>
inline T parallel_it_init(T start, Q& x, const R& X, Args&&... tuple) {
    start = parallel_it_init(start, static_cast<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool parallel_it_step() {
    return true;
}
template <typename Q, typename R, typename... Args>
inline bool parallel_it_step(Q& x, const R& X, Args&&... tuple) {
    if (parallel_it_step(static_cast<Args>(tuple)...)) {
        if (++x - X == 0) {
            x = 0;
            return true;
        }
    }
    return false;
}

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
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

namespace helpers {
template <typename T>
struct NumOfLambdaArgs : public NumOfLambdaArgs<decltype(&T::operator())> {};

template <typename C, typename R, typename... Args>
struct NumOfLambdaArgs<R (C::*)(Args...) const> {
    constexpr static int value = sizeof...(Args);
};

template <typename ACT, typename... T, size_t N_ARGS = NumOfLambdaArgs<ACT>::value>
typename std::enable_if<N_ARGS == sizeof...(T) + 2, void>::type call_with_args(const ACT& body,
                                                                               size_t g_id,
                                                                               size_t iwork,
                                                                               T... arg) {
    body(g_id, iwork, arg...);
}

template <typename ACT, typename... T, size_t N_ARGS = NumOfLambdaArgs<ACT>::value>
typename std::enable_if<N_ARGS == sizeof...(T) + 1, void>::type call_with_args(const ACT& body,
                                                                               size_t g_id,
                                                                               size_t iwork,
                                                                               T... arg) {
    body(g_id, arg...);
}

template <typename ACT, typename... T, size_t N_ARGS = NumOfLambdaArgs<ACT>::value>
typename std::enable_if<N_ARGS == sizeof...(T), void>::type call_with_args(const ACT& body,
                                                                           size_t g_id,
                                                                           size_t iwork,
                                                                           T... arg) {
    body(arg...);
}
}  // namespace helpers

template <typename T0, typename F>
void for_1d(const int& ithr, const int& nthr, const T0& D0, const F& func) {
    T0 d0{0}, end{0};
    splitter(D0, nthr, ithr, d0, end);
    for (; d0 < end; ++d0)
        helpers::call_with_args(func, ithr, d0, d0);
}

template <typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
    auto work_amount = static_cast<size_t>(D0);
    int nthr = static_cast<int>(getTotalThreads());
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_1d(0, 1, D0, func);
    } else {
        TrySimpleParallelFor(static_cast<size_t>(nthr), [&](size_t ithr) {
            for_1d(static_cast<int>(ithr), nthr, D0, func);
        });
    }
}

template <typename T0, typename T1, typename F>
void for_2d(const int& ithr, const int& nthr, const T0& D0, const T1& D1, const F& func) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0)
        return;
    size_t start{0}, end{0};
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{0};
    T1 d1{0};
    parallel_it_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        helpers::call_with_args(func, ithr, iwork, d0, d1);
        parallel_it_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename F>
void parallel_for2d(const T0& D0, const T1& D1, const F& func) {
    auto work_amount = static_cast<size_t>(D0 * D1);
    int nthr = static_cast<int>(getTotalThreads());
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_2d(0, 1, D0, D1, func);
    } else {
        TrySimpleParallelFor(static_cast<size_t>(nthr), [&](size_t ithr) {
            for_2d(static_cast<int>(ithr), nthr, D0, D1, func);
        });
    }
}

template <typename T0, typename T1, typename T2, typename F>
void for_3d(const int& ithr, const int& nthr, const T0& D0, const T1& D1, const T2& D2, const F& func) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0)
        return;
    size_t start{0}, end{0};
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{0};
    T1 d1{0};
    T2 d2{0};
    parallel_it_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        helpers::call_with_args(func, ithr, iwork, d0, d1, d2);
        parallel_it_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_for3d(const T0& D0, const T1& D1, const T2& D2, const F& func) {
    auto work_amount = static_cast<size_t>(D0 * D1 * D2);
    int nthr = static_cast<int>(getTotalThreads());
    if (static_cast<size_t>(nthr) > work_amount)
        nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        for_3d(0, 1, D0, D1, D2, func);
    } else {
        TrySimpleParallelFor(static_cast<size_t>(nthr), [&](size_t ithr) {
            for_3d(static_cast<int>(ithr), nthr, D0, D1, D2, func);
        });
    }
}

};  // namespace cpu
};  // namespace ov