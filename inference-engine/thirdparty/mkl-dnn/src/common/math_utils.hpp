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

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <stdint.h>
#include <math.h>

#include "utils.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace math {

template <typename data_t, typename acc_t>
inline typename utils::enable_if<!nstl::is_integral<data_t>::value,
       typename utils::remove_reference<data_t>::type>::type
saturate(const acc_t &x) { return x; }

template <typename data_t, typename acc_t>
inline typename utils::enable_if<nstl::is_integral<data_t>::value,
       typename utils::remove_reference<data_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    if (v < (acc_t)nstl::numeric_limits<data_t>::lowest())
        v = (acc_t)nstl::numeric_limits<data_t>::lowest();
    if (v > (acc_t)nstl::numeric_limits<data_t>::max())
        v = (acc_t)nstl::numeric_limits<data_t>::max();
    return (typename utils::remove_reference<data_t>::type)v;
}

template <> inline int8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <typename out_t>
inline typename utils::enable_if<nstl::is_integral<out_t>::value, out_t>::type
out_round(float v, round_mode_t rmode = round_mode::nearest)
{ return (out_t)(rmode == round_mode::down ? floorf(v) : nearbyintf(v)); }

template <typename out_t>
inline typename utils::enable_if<!nstl::is_integral<out_t>::value, out_t>::type
out_round(float v, round_mode_t rmode = round_mode::nearest)
{ UNUSED(rmode); return v; }

inline int gcd(int a, int b) {
	a = impl::nstl::abs(a);
	b = impl::nstl::abs(b);
	if (a < b) { int x = a; a = b; b = x; }

	if (b == 0) return a;

	int r;
	while ((r = a % b) != 0) { a = b; b = r; }

	return b;
}

template <typename T>
inline bool is_pow2(const T& v) { return (v & (v - 1)) == 0; }

/** returns floor(log2(v)), aka the position of the leftmost non-0 bit */
inline int ilog2q(size_t v) {
    if (v == 0)
        return -1;

    int p = 0;
#   define CP(pw) do { if (v >= (1ull << pw)) { v >>= pw; p += pw; } } while(0)
    CP(32); CP(16); CP(8); CP(4); CP(2); CP(1);
#   undef CP
    return p;
}

/* activation */
template <typename T, typename A> inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : (T)(s * alpha);
}
template <typename T, typename A> inline T relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : (T)(dd * alpha);
}

template <typename T> inline T tanh_fwd(T s) {
    const float e = tanhf((float) s);
    return (T) e;
}
template <typename T> inline T tanh_bwd(T dd, T s) {
    const float e = tanh_fwd<float>((float) s);
    return (T)(dd * (1 - e * e));
}

template <typename T, typename A> inline T elu_fwd(T s, A alpha) {
    return s > 0 ? s : (T)(alpha * (::expm1f((float)s)));
}
template <typename T, typename A> inline T elu_bwd(T dd, T s, A alpha) {
    return (T)(dd * (s > 0 ? 1 : alpha * ::expf((float)s)));
}

template <typename T>
inline T square_fwd(T s) {
    return s * s;
}

template <typename T>
inline T square_bwd(T dd, T s) {
    return dd * 2*s;
}

template <typename T>
inline T abs_fwd(T s) {
    return s > 0 ? s : -s;
}

template <typename T>
inline T abs_bwd(T dd, T s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

template <typename T>
inline T sqrt_fwd(T s) {
    return s > 0 ? (T)(::sqrtf((float)(s))) : 0;
}

template <typename T>
inline T sqrt_bwd(T dd, T s) {
    return s > 0
        ? (T)(dd / (2 * ::sqrtf((float)(s))))
        : 0;
}

template <typename T, typename A>
inline T linear_fwd(T s, A alpha, A beta) {
    return (T)(alpha * s + beta);
}

template <typename T, typename A>
inline T linear_bwd(T dd, T s, A alpha, A beta) {
    (void) s;
    (void) beta;
    return (T)(dd * alpha);
}

template <typename T, typename A>
inline T bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? (T)(alpha) : s;
}

template <typename T, typename A>
inline T bounded_relu_bwd(T dd, T s, A alpha) {
    return dd * (0 < s && s < alpha ? 1 : 0);
}

template <typename T>
inline T soft_relu_fwd(T s) {
    return (T)(::logf(1 + ::expf((float)s)));
}

template <typename T>
inline T soft_relu_bwd(T dd, T s) {
    return (T)(dd / (1 + ::expf((float)(-s))));
}

template <typename T>
inline T logistic_fwd(T s) {
    T v = (T)(::tanhf((float) s/2));
    return (1 + v)/2;
}

template <typename T>
inline T logistic_bwd(T dd, T s) {
    T v = logistic_fwd<T>(s);
    return dd * v * (1 - v);
}

}
}
}

#endif
