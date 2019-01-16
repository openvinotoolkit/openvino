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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "c_types_map.hpp"

namespace mkldnn {
namespace impl {

#define UNUSED(x) ((void)x)
#define MAYBE_UNUSED(x) UNUSED(x)

#define CHECK(f) do { \
    status_t status = f; \
    if (status != status::success) \
    return status; \
} while (0)

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#ifdef __APPLE__
// older XCode doesn't support thread_local
#define THREAD_LOCAL __thread
#else
#define THREAD_LOCAL thread_local
#endif

namespace utils {

/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for mkl-dnn
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */
template<bool expr, class T = void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };

/* analogue std::conditional */
template <bool, typename, typename> struct conditional {};
template <typename T, typename F> struct conditional<true, T, F>
{ typedef T type; };
template <typename T, typename F> struct conditional<false, T, F>
{ typedef F type; };

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> { typedef T type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> { typedef FT type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> { typedef FF type; };

template <bool, typename U, U, U> struct conditional_v {};
template <typename U, U t, U f> struct conditional_v<true, U, t, f>
{ static constexpr U value = t; };
template <typename U, U t, U f> struct conditional_v<false, U, t, f>
{ static constexpr U value = f; };

template <typename T> struct remove_reference { typedef T type; };
template <typename T> struct remove_reference<T&> { typedef T type; };
template <typename T> struct remove_reference<T&&> { typedef T type; };

template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &t)
{ return static_cast<T&&>(t); }
template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &&t)
{ return static_cast<T&&>(t); }

template <typename T>
inline typename remove_reference<T>::type zero()
{ auto zero = typename remove_reference<T>::type(); return zero; }

template <typename T, typename P>
inline bool everyone_is(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) { return one_of(nullptr, ptrs...); }

inline bool implication(bool cause, bool effect) { return !cause || effect; }

template<typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}
template<typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) return false;
    return true;
}
template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = static_cast<T>(val);
}

namespace product_impl {
template<size_t> struct int2type{};

template <typename T>
constexpr int product_impl(const T *arr, int2type<0>) { return arr[0]; }

template <typename T, size_t num>
inline T product_impl(const T *arr, int2type<num>) {
    return arr[0]*product_impl(arr+1, int2type<num-1>()); }
}

template <size_t num, typename T>
inline T array_product(const T *arr) {
    return product_impl::product_impl(arr, product_impl::int2type<num-1>());
}

template<typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    R prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_dn(const T a, const U b) {
    return (a / b) * b;
}

template <typename T, typename U, typename V>
inline U this_block_size(const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;
    if (block_boundary > max)
        return max - offset;
    else
        return block_size;
}

template<typename T>
inline T nd_iterator_init(T start) { return start; }
template<typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&... tuple) {
    start = nd_iterator_init(start, utils::forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() { return true; }
template<typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&... tuple) {
    if (nd_iterator_step(utils::forward<Args>(tuple)...) ) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template<typename U, typename W, typename Y>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X)
{
    U max_jump = end - cur;
    U dim_jump = X - x;
    if (dim_jump <= max_jump) {
        x = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x += max_jump;
        return false;
    }
}
template<typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X,
        Args &&... tuple)
{
    if (nd_iterator_jump(cur, end, utils::forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs) : _dims{ Fargs... }
    {
        _base_ptr = base;
    }
    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs)
    {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element)
    {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element)
    {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs)
    {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem *_base_ptr;
    const int _dims[Tdims];
};

}

void *malloc(size_t size, int alignment);
void free(void *p);

struct c_compatible {
    enum { default_alignment = 64 };
    static void *operator new(size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void *operator new(size_t sz, void *p) { UNUSED(sz); return p; }
    static void *operator new[](size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void operator delete(void *p) { free(p); }
    static void operator delete[](void *p) { free(p); }
};

inline void yield_thread() { }

int mkldnn_getenv(char *value, const char *name, int len);
bool mkldnn_jit_dump();
FILE *mkldnn_fopen(const char *filename, const char *mode);

#ifdef MKLDNN_JIT
void set_rnd_mode(round_mode_t rnd_mode);
void restore_rnd_mode();
#endif

unsigned int get_cache_size(int level, bool per_core);

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
