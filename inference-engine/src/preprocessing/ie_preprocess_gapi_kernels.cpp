// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"

#if CPU_SIMD
  #include "ie_system_conf.h"

#ifdef HAVE_AVX512
  #include "cpu_x86_avx512/ie_preprocess_gapi_kernels_avx512.hpp"
#endif

#ifdef HAVE_AVX2
  #include "cpu_x86_avx2/ie_preprocess_gapi_kernels_avx2.hpp"
#endif

#ifdef HAVE_SSE
  #include "cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.hpp"
#endif

#endif

#ifdef HAVE_NEON
  #include "arm_neon/ie_preprocess_gapi_kernels_neon.hpp"
#endif

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>
#include <opencv2/gapi/gcompoundkernel.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>
#include <functional>

#if defined(__GNUC__) && (__GNUC__ <= 5)
#include <cmath>
#endif

namespace InferenceEngine {
namespace gapi {
namespace kernels {

template<typename T, int chs> static
void mergeRow(const std::array<const uint8_t*, chs>& ins, uint8_t* out, int length) {
// AVX512 implementation of wide universal intrinsics is slower than AVX2.
// It is turned off until the cause isn't found out.
#if 0
#ifdef HAVE_AVX512
    if (with_cpu_x86_avx512f()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            avx512::mergeRow_8UC2(ins[0], ins[1], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            avx512::mergeRow_8UC3(ins[0], ins[1], ins[2], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            avx512::mergeRow_8UC4(ins[0], ins[1], ins[2], ins[3], out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            avx512::mergeRow_32FC2(reinterpret_cast<const float*>(ins[0]),
                                   reinterpret_cast<const float*>(ins[1]),
                                   reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            avx512::mergeRow_32FC3(reinterpret_cast<const float*>(ins[0]),
                                   reinterpret_cast<const float*>(ins[1]),
                                   reinterpret_cast<const float*>(ins[2]),
                                   reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            avx512::mergeRow_32FC4(reinterpret_cast<const float*>(ins[0]),
                                   reinterpret_cast<const float*>(ins[1]),
                                   reinterpret_cast<const float*>(ins[2]),
                                   reinterpret_cast<const float*>(ins[3]),
                                   reinterpret_cast<float*>(out), length);
            return;
        }
    }
#endif  // HAVE_AVX512
#endif

#ifdef HAVE_AVX2
    if (with_cpu_x86_avx2()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            avx::mergeRow_8UC2(ins[0], ins[1], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            avx::mergeRow_8UC3(ins[0], ins[1], ins[2], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            avx::mergeRow_8UC4(ins[0], ins[1], ins[2], ins[3], out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            avx::mergeRow_32FC2(reinterpret_cast<const float*>(ins[0]),
                                reinterpret_cast<const float*>(ins[1]),
                                reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            avx::mergeRow_32FC3(reinterpret_cast<const float*>(ins[0]),
                                reinterpret_cast<const float*>(ins[1]),
                                reinterpret_cast<const float*>(ins[2]),
                                reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            avx::mergeRow_32FC4(reinterpret_cast<const float*>(ins[0]),
                                reinterpret_cast<const float*>(ins[1]),
                                reinterpret_cast<const float*>(ins[2]),
                                reinterpret_cast<const float*>(ins[3]),
                                reinterpret_cast<float*>(out), length);
            return;
        }
    }
#endif  // HAVE_AVX2

#ifdef HAVE_SSE
    if (with_cpu_x86_sse42()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            mergeRow_8UC2(ins[0], ins[1], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            mergeRow_8UC3(ins[0], ins[1], ins[2], out, length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            mergeRow_8UC4(ins[0], ins[1], ins[2], ins[3], out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            mergeRow_32FC2(reinterpret_cast<const float*>(ins[0]),
                           reinterpret_cast<const float*>(ins[1]),
                           reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            mergeRow_32FC3(reinterpret_cast<const float*>(ins[0]),
                           reinterpret_cast<const float*>(ins[1]),
                           reinterpret_cast<const float*>(ins[2]),
                           reinterpret_cast<float*>(out), length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            mergeRow_32FC4(reinterpret_cast<const float*>(ins[0]),
                           reinterpret_cast<const float*>(ins[1]),
                           reinterpret_cast<const float*>(ins[2]),
                           reinterpret_cast<const float*>(ins[3]),
                           reinterpret_cast<float*>(out), length);
            return;
        }
    }
#endif  // HAVE_SSE

#ifdef HAVE_NEON
    if (std::is_same<T, uint8_t>::value && chs == 2) {
        neon::mergeRow_8UC2(ins[0], ins[1], out, length);
        return;
    }

    if (std::is_same<T, uint8_t>::value && chs == 3) {
        neon::mergeRow_8UC3(ins[0], ins[1], ins[2], out, length);
        return;
    }

    if (std::is_same<T, uint8_t>::value && chs == 4) {
        neon::mergeRow_8UC4(ins[0], ins[1], ins[2], ins[3], out, length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 2) {
        neon::mergeRow_32FC2(reinterpret_cast<const float*>(ins[0]),
                             reinterpret_cast<const float*>(ins[1]),
                             reinterpret_cast<float*>(out), length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 3) {
        neon::mergeRow_32FC3(reinterpret_cast<const float*>(ins[0]),
                             reinterpret_cast<const float*>(ins[1]),
                             reinterpret_cast<const float*>(ins[2]),
                             reinterpret_cast<float*>(out), length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 4) {
        neon::mergeRow_32FC4(reinterpret_cast<const float*>(ins[0]),
                             reinterpret_cast<const float*>(ins[1]),
                             reinterpret_cast<const float*>(ins[2]),
                             reinterpret_cast<const float*>(ins[3]),
                             reinterpret_cast<float*>(out), length);
        return;
    }
#endif  // HAVE_NEON

    const T* insT[chs];
    for (int c = 0; c < chs; c++) {
        insT[c] = reinterpret_cast<const T*>(ins[c]);
    }
    auto outT = reinterpret_cast<T*>(out);

    for (int x = 0; x < length; x++) {
        for (int c = 0; c < chs; c++) {
            outT[chs*x + c] = insT[c][x];
        }
    }
}

template<typename T, int chs> static
void splitRow(const uint8_t* in, std::array<uint8_t*, chs>& outs, int length) {
#ifdef HAVE_AVX512
    if (with_cpu_x86_avx512f()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            avx512::splitRow_8UC2(in, outs[0], outs[1], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            avx512::splitRow_8UC3(in, outs[0], outs[1], outs[2], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            avx512::splitRow_8UC4(in, outs[0], outs[1], outs[2], outs[3], length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            avx512::splitRow_32FC2(reinterpret_cast<const float*>(in),
                                   reinterpret_cast<float*>(outs[0]),
                                   reinterpret_cast<float*>(outs[1]),
                                   length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            avx512::splitRow_32FC3(reinterpret_cast<const float*>(in),
                                   reinterpret_cast<float*>(outs[0]),
                                   reinterpret_cast<float*>(outs[1]),
                                   reinterpret_cast<float*>(outs[2]),
                                   length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            avx512::splitRow_32FC4(reinterpret_cast<const float*>(in),
                                   reinterpret_cast<float*>(outs[0]),
                                   reinterpret_cast<float*>(outs[1]),
                                   reinterpret_cast<float*>(outs[2]),
                                   reinterpret_cast<float*>(outs[3]),
                                   length);
            return;
        }
    }
#endif  // HAVE_AVX512

#ifdef HAVE_AVX2

    if (with_cpu_x86_avx2()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            avx::splitRow_8UC2(in, outs[0], outs[1], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            avx::splitRow_8UC3(in, outs[0], outs[1], outs[2], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            avx::splitRow_8UC4(in, outs[0], outs[1], outs[2], outs[3], length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            avx::splitRow_32FC2(reinterpret_cast<const float*>(in),
                                reinterpret_cast<float*>(outs[0]),
                                reinterpret_cast<float*>(outs[1]),
                                length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            avx::splitRow_32FC3(reinterpret_cast<const float*>(in),
                                reinterpret_cast<float*>(outs[0]),
                                reinterpret_cast<float*>(outs[1]),
                                reinterpret_cast<float*>(outs[2]),
                                length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            avx::splitRow_32FC4(reinterpret_cast<const float*>(in),
                                reinterpret_cast<float*>(outs[0]),
                                reinterpret_cast<float*>(outs[1]),
                                reinterpret_cast<float*>(outs[2]),
                                reinterpret_cast<float*>(outs[3]),
                                length);
            return;
        }
    }
#endif  // HAVE_AVX2

#ifdef HAVE_SSE
    if (with_cpu_x86_sse42()) {
        if (std::is_same<T, uint8_t>::value && chs == 2) {
            splitRow_8UC2(in, outs[0], outs[1], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 3) {
            splitRow_8UC3(in, outs[0], outs[1], outs[2], length);
            return;
        }

        if (std::is_same<T, uint8_t>::value && chs == 4) {
            splitRow_8UC4(in, outs[0], outs[1], outs[2], outs[3], length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 2) {
            splitRow_32FC2(reinterpret_cast<const float*>(in),
                           reinterpret_cast<float*>(outs[0]),
                           reinterpret_cast<float*>(outs[1]),
                           length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 3) {
            splitRow_32FC3(reinterpret_cast<const float*>(in),
                           reinterpret_cast<float*>(outs[0]),
                           reinterpret_cast<float*>(outs[1]),
                           reinterpret_cast<float*>(outs[2]),
                           length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 4) {
            splitRow_32FC4(reinterpret_cast<const float*>(in),
                           reinterpret_cast<float*>(outs[0]),
                           reinterpret_cast<float*>(outs[1]),
                           reinterpret_cast<float*>(outs[2]),
                           reinterpret_cast<float*>(outs[3]),
                           length);
            return;
        }
    }
#endif  // HAVE_SSE

#ifdef HAVE_NEON
    if (std::is_same<T, uint8_t>::value && chs == 2) {
        neon::splitRow_8UC2(in, outs[0], outs[1], length);
        return;
    }

    if (std::is_same<T, uint8_t>::value && chs == 3) {
        neon::splitRow_8UC3(in, outs[0], outs[1], outs[2], length);
        return;
    }

    if (std::is_same<T, uint8_t>::value && chs == 4) {
        neon::splitRow_8UC4(in, outs[0], outs[1], outs[2], outs[3], length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 2) {
        neon::splitRow_32FC2(reinterpret_cast<const float*>(in),
                             reinterpret_cast<float*>(outs[0]),
                             reinterpret_cast<float*>(outs[1]),
                             length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 3) {
        neon::splitRow_32FC3(reinterpret_cast<const float*>(in),
                             reinterpret_cast<float*>(outs[0]),
                             reinterpret_cast<float*>(outs[1]),
                             reinterpret_cast<float*>(outs[2]),
                             length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 4) {
        neon::splitRow_32FC4(reinterpret_cast<const float*>(in),
                             reinterpret_cast<float*>(outs[0]),
                             reinterpret_cast<float*>(outs[1]),
                             reinterpret_cast<float*>(outs[2]),
                             reinterpret_cast<float*>(outs[3]),
                             length);
        return;
    }
#endif  // HAVE_NEON

    auto inT = reinterpret_cast<const T*>(in);

    T* outsT[chs];
    for (int c = 0; c < chs; c++) {
        outsT[c] = reinterpret_cast<T*>(outs[c]);
    }

    for (int x = 0; x < length; x++) {
        for (int c = 0; c < chs; c++) {
            outsT[c][x] = inT[chs*x + c];
        }
    }
}

namespace {

struct fp_16_t {
    int16_t v;
};


template<typename type>
struct cv_type_to_depth;

template<> struct cv_type_to_depth<std::uint8_t>    { enum { depth = CV_8U  }; };
template<> struct cv_type_to_depth<std::int8_t>     { enum { depth = CV_8S  }; };
template<> struct cv_type_to_depth<std::uint16_t>   { enum { depth = CV_16U }; };
template<> struct cv_type_to_depth<std::int16_t>    { enum { depth = CV_16S }; };
template<> struct cv_type_to_depth<std::int32_t>    { enum { depth = CV_32S }; };
template<> struct cv_type_to_depth<float>           { enum { depth = CV_32F }; };
template<> struct cv_type_to_depth<fp_16_t>         { enum { depth = CV_16F }; };

template<typename ... types>
struct typelist {};

template<typename type_list>
struct head;

template<template<typename ...> class list, typename head_t, typename ... types>
struct head<list<head_t, types...>> { using type = head_t;};

template<typename typelist>
using head_t = typename head<typelist>::type;

template<typename type>
struct type_to_type {};

template <typename typelist>
struct type_dispatch_impl;

template <template<typename ...> class typelist, typename... type>
struct type_dispatch_impl<typelist<type...>> {
    template <typename result_t, typename default_t, typename type_id_t, typename type_to_id_t, typename type_to_value_t>
    static result_t dispatch(type_id_t type_id, type_to_id_t&& type_to_id, type_to_value_t&& type_to_value, default_t default_value) {
        result_t res = default_value;

        std::initializer_list<int> ({(type_id == type_to_id(type_to_type<type>{}) ? (res = type_to_value(type_to_type<type>{})), 0 : 0)...});
        return res;
    }
};

}  // namespace

template <typename typelist, typename default_t, typename type_id_t, typename type_to_id_t, typename type_to_value_t,
          typename result_t = decltype(std::declval<type_to_value_t>()(type_to_type<head_t<typelist>> {}))>
result_t type_dispatch(type_id_t type_id, type_to_id_t&& type_to_id, type_to_value_t&& type_to_value, default_t default_value = {}) {
    return type_dispatch_impl<typelist>::template dispatch<result_t>(std::forward<type_id_t>(type_id),
                                                                     std::forward<type_to_id_t>(type_to_id),
                                                                     std::forward<type_to_value_t>(type_to_value),
                                                                     std::forward<default_t>(default_value));
}

namespace {

struct cv_type_id {
    template <typename type>
    const int operator()(type_to_type<type> ) { return cv_type_to_depth<type>::depth;}
};

}  // namespace

template <typename typelist>
bool is_cv_type_in_list(const int type_id) {
    return type_dispatch<typelist>(type_id, cv_type_id{}, [](...){ return true;}, false);
}

namespace {

using merge_supported_types = typelist<uint8_t, int8_t, uint16_t, int16_t, int32_t, float, fp_16_t>;

template<int chs>
struct typed_merge_row {
    using p_f = void (*)(const std::array<const uint8_t*, chs>& ins, uint8_t* out, int length);

    template <typename type>
    p_f operator()(type_to_type<type> ) { return mergeRow<type, chs>; }

    p_f operator()(type_to_type<fp_16_t> ) {
        static_assert(sizeof(fp_16_t) == sizeof(fp_16_t::v),
                "fp_16_t should be a plain wrap over FP16 implementation type");
        return mergeRow<decltype(fp_16_t::v), chs>;
    }
};

}  // namespace

GAPI_FLUID_KERNEL(FMerge2, Merge2, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View& a,
                    const cv::gapi::fluid::View& b,
                          cv::gapi::fluid::Buffer& out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{}, typed_merge_row<2>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({a.InLineB(l), b.InLineB(l)}, out.OutLineB(l), a.length());
        }
    }
};

GAPI_FLUID_KERNEL(FMerge3, Merge3, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View& a,
                    const cv::gapi::fluid::View& b,
                    const cv::gapi::fluid::View& c,
                          cv::gapi::fluid::Buffer& out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{}, typed_merge_row<3>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({a.InLineB(l), b.InLineB(l), c.InLineB(l)}, out.OutLineB(l), a.length());
        }
    }
};

GAPI_FLUID_KERNEL(FMerge4, Merge4, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View& a,
                    const cv::gapi::fluid::View& b,
                    const cv::gapi::fluid::View& c,
                    const cv::gapi::fluid::View& d,
                          cv::gapi::fluid::Buffer& out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{}, typed_merge_row<4>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({a.InLineB(l), b.InLineB(l), c.InLineB(l), d.InLineB(l)}, out.OutLineB(l), a.length());
        }
    }
};


namespace {
using split_supported_types = typelist<uint8_t, int8_t, uint16_t, int16_t, int32_t, float, fp_16_t>;

template<int chs>
struct typed_split_row {
    using p_f = void (*)(const uint8_t* in, std::array<uint8_t*, chs>& outs, int length);

    template <typename type>
    p_f operator()(type_to_type<type> ) { return splitRow<type, chs>; }

    p_f operator()(type_to_type<fp_16_t> ) {
        static_assert(sizeof(fp_16_t) == sizeof(fp_16_t::v),
                "fp_16_t should be a plain wrap over FP16 implementation type");
        return splitRow<decltype(fp_16_t::v), chs>;
    }
};

}  // namespace

GAPI_FLUID_KERNEL(FSplit2, Split2, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View  & in,
                          cv::gapi::fluid::Buffer& out1,
                          cv::gapi::fluid::Buffer& out2) {
        GAPI_DbgAssert(2 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{}, typed_split_row<2>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 2> outs = {out1.OutLineB(i), out2.OutLineB(i)};
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

GAPI_FLUID_KERNEL(FSplit3, Split3, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View  & in,
                          cv::gapi::fluid::Buffer& out1,
                          cv::gapi::fluid::Buffer& out2,
                          cv::gapi::fluid::Buffer& out3) {
        GAPI_DbgAssert(3 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(1 == out3.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out3.meta().depth);

        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{}, typed_split_row<3>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 3> outs = {out1.OutLineB(i), out2.OutLineB(i),
                                            out3.OutLineB(i)};
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

GAPI_FLUID_KERNEL(FSplit4, Split4, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View  & in,
                          cv::gapi::fluid::Buffer& out1,
                          cv::gapi::fluid::Buffer& out2,
                          cv::gapi::fluid::Buffer& out3,
                          cv::gapi::fluid::Buffer& out4) {
        GAPI_DbgAssert(4 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(1 == out3.meta().chan);
        GAPI_DbgAssert(1 == out4.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out3.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out4.meta().depth);
        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{}, typed_split_row<4>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 4> outs = {out1.OutLineB(i), out2.OutLineB(i),
                                            out3.OutLineB(i), out4.OutLineB(i)};
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

//----------------------------------------------------------------------

template<typename T>
static void chanToPlaneRow(const uint8_t* in, int chan, int chs, uint8_t* out, int length) {
// AVX512 implementation of wide universal intrinsics is slower than AVX2.
// It is turned off until the cause isn't found out.
#if 0
    #ifdef HAVE_AVX512
    if (with_cpu_x86_avx512f()) {
        if (std::is_same<T, uint8_t>::value && chs == 1) {
            avx512::copyRow_8U(in, out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 1) {
            avx512::copyRow_32F(reinterpret_cast<const float*>(in),
                                reinterpret_cast<float*>(out),
                                length);
            return;
        }
    }
    #endif  // HAVE_AVX512
#endif

    #ifdef HAVE_AVX2
    if (with_cpu_x86_avx2()) {
        if (std::is_same<T, uint8_t>::value && chs == 1) {
            avx::copyRow_8U(in, out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 1) {
            avx::copyRow_32F(reinterpret_cast<const float*>(in),
                             reinterpret_cast<float*>(out),
                             length);
            return;
        }
    }
    #endif  // HAVE_AVX2
    #ifdef HAVE_SSE
    if (with_cpu_x86_sse42()) {
        if (std::is_same<T, uint8_t>::value && chs == 1) {
            copyRow_8U(in, out, length);
            return;
        }

        if (std::is_same<T, float>::value && chs == 1) {
            copyRow_32F(reinterpret_cast<const float*>(in),
                        reinterpret_cast<float*>(out),
                        length);
            return;
        }
    }
    #endif  // HAVE_SSE

    #ifdef HAVE_NEON
    if (std::is_same<T, uint8_t>::value && chs == 1) {
        neon::copyRow_8U(in, out, length);
        return;
    }

    if (std::is_same<T, float>::value && chs == 1) {
        neon::copyRow_32F(reinterpret_cast<const float*>(in),
                          reinterpret_cast<float*>(out),
                          length);
        return;
    }
    #endif  // HAVE_NEON

    const auto inT  = reinterpret_cast<const T*>(in);
          auto outT = reinterpret_cast<      T*>(out);

    for (int x = 0; x < length; x++) {
        outT[x] = inT[x*chs + chan];
    }
}

//    GAPI_OCV_KERNEL(OCVChanToPlane, ChanToPlane) {
//        static void run(const cv::Mat &in, int chan, cv::Mat &out) {
//            out.create(in.rows, in.cols, in.depth());
//            const auto rowFunc = (in.depth() == CV_8U) ? &chanToPlaneRow<uint8_t> : &chanToPlaneRow<float>;

//            for (int y = 0; y < out.rows; y++)
//            {
//                rowFunc(in.data + y*in.step, chan, in.channels(), out.data + y*out.step, in.cols);
//            }
//        }
//    };

//    GAPI_OCV_KERNEL(OCVScalePlane, ScalePlane) {
//        static void run(const cv::Mat &in, int /*type*/, const Size &sz, int interp, cv::Mat &out) {
//            cv::resize(in, out, sz, 0, 0, interp);
//        }
//    };

//    GAPI_OCV_KERNEL(OCVMerge2, Merge2) {
//        static void run(const cv::Mat &a, const cv::Mat &b, cv::Mat out) {
//            out.create(a.rows, a.cols, CV_MAKETYPE(a.depth(), 2));
//            const auto rowFunc = (a.depth() == CV_8U) ? &mergeRow<uint8_t, 2> : &mergeRow<float, 2>;

//            for (int y = 0; y < out.rows; y++)
//            {
//                rowFunc({a.data + y*a.step, b.data + y*b.step}, out.data + out.step, a.cols);
//            }
//        }
//    };

GAPI_FLUID_KERNEL(FChanToPlane, ChanToPlane, false) {
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View& in, int chan,
                    cv::gapi::fluid::Buffer& out) {
        const auto rowFunc = (in.meta().depth == CV_8U) ? &chanToPlaneRow<uint8_t> : &chanToPlaneRow<float>;
        rowFunc(in.InLineB(0), chan, in.meta().chan, out.OutLineB(), in.length());
    }
};

//----------------------------------------------------------------------

G_TYPED_KERNEL(ScalePlane8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_8u") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(ScalePlane32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_32f") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(UpscalePlaneArea8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.upscale_plane_area_8u") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
        GAPI_DbgAssert(in.size.width < sz.width || in.size.height < sz.height);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(UpscalePlaneArea32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.upscale_plane_area_32f") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
        GAPI_DbgAssert(in.size.width < sz.width || in.size.height < sz.height);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(ScalePlaneArea8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_area_8u") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
        GAPI_DbgAssert(in.size.width >= sz.width && in.size.height >= sz.height);
        return in.withSize(sz);
    }
};

G_TYPED_KERNEL(ScalePlaneArea32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_area_32f") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in, const Size &sz, int) {
        GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
        GAPI_DbgAssert(in.size.width >= sz.width && in.size.height >= sz.height);
        return in.withSize(sz);
    }
};

GAPI_COMPOUND_KERNEL(FScalePlane, ScalePlane) {
    static cv::GMat expand(cv::GMat in, int type, const Size& szIn, const Size& szOut, int interp) {
        GAPI_DbgAssert(CV_8UC1 == type || CV_32FC1 == type);
        GAPI_DbgAssert(cv::INTER_AREA == interp || cv::INTER_LINEAR == interp);

        if (cv::INTER_AREA == interp) {
            bool upscale = szIn.width < szOut.width || szIn.height < szOut.height;
            if (CV_8UC1 == type) {
                if (upscale)
                    return UpscalePlaneArea8u::on(in, szOut, interp);
                else
                    return   ScalePlaneArea8u::on(in, szOut, interp);
            }
            if (CV_32FC1 == type) {
                if (upscale)
                    return UpscalePlaneArea32f::on(in, szOut, interp);
                else
                    return   ScalePlaneArea32f::on(in, szOut, interp);
            }
        }

        if (cv::INTER_LINEAR == interp) {
            if (CV_8UC1 == type) {
                return ScalePlane8u::on(in, szOut, interp);
            }
            if (CV_32FC1 == type) {
                return ScalePlane32f::on(in, szOut, interp);
            }
        }

        GAPI_Assert(!"unsupported parameters");
        return {};
    }
};

static inline double invRatio(int inSz, int outSz) {
    return static_cast<double>(outSz) / inSz;
}

static inline double ratio(int inSz, int outSz) {
    return 1 / invRatio(inSz, outSz);
}

template<typename T, typename Mapper, int chanNum>
struct linearScratchDesc {
    using alpha_t = typename Mapper::alpha_type;
    using index_t = typename Mapper::index_type;

    alpha_t* alpha;
    alpha_t* clone;
    index_t* mapsx;
    alpha_t* beta;
    index_t* mapsy;
    T*       tmp;

    linearScratchDesc(int /*inW*/, int /*inH*/, int outW, int outH,  void* data) {
        alpha = reinterpret_cast<alpha_t*>(data);
        clone = reinterpret_cast<alpha_t*>(alpha + outW);
        mapsx = reinterpret_cast<index_t*>(clone + outW*4);
        beta  = reinterpret_cast<alpha_t*>(mapsx + outW);
        mapsy = reinterpret_cast<index_t*>(beta  + outH);
        tmp   = reinterpret_cast<T*>      (mapsy + outH*2);
    }

    static int bufSize(int inW, int /*inH*/, int outW, int outH, int lpi) {
        auto size = outW * sizeof(alpha_t)     +
                    outW * sizeof(alpha_t) * 4 +  // alpha clones // previous alpha is redundant?
                    outW * sizeof(index_t)     +
                    outH * sizeof(alpha_t)     +
                    outH * sizeof(index_t) * 2 +
                     inW * sizeof(T) * lpi * chanNum;

        return static_cast<int>(size);
    }
};

template<typename T, typename Mapper, int chanNum = 1>
static void initScratchLinear(const cv::GMatDesc& in,
                              const         Size& outSz,
                              cv::gapi::fluid::Buffer& scratch,
                                             int  lpi) {
    using alpha_type = typename Mapper::alpha_type;
    static const auto unity = Mapper::unity;

    auto inSz = in.size;
    auto sbufsize = linearScratchDesc<T, Mapper, chanNum>::bufSize(inSz.width, inSz.height, outSz.width, outSz.height, lpi);

    Size scratch_size{sbufsize, 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    double hRatio = ratio(in.size.width, outSz.width);
    double vRatio = ratio(in.size.height, outSz.height);

    linearScratchDesc<T, Mapper, chanNum> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    auto *alpha = scr.alpha;
    auto *clone = scr.clone;
    auto *index = scr.mapsx;

    for (int x = 0; x < outSz.width; x++) {
        auto map = Mapper::map(hRatio, 0, in.size.width, x);
        auto alpha0 = map.alpha0;
        auto index0 = map.index0;

        // TRICK:
        // Algorithm takes pair of input pixels, sx0'th and sx1'th,
        // and compute result as alpha0*src[sx0] + alpha1*src[sx1].
        // By definition: sx1 == sx0 + 1 either sx1 == sx0, and
        // alpha0 + alpha1 == unity (scaled appropriately).
        // Here we modify formulas for alpha0 and sx1: by assuming
        // that sx1 == sx0 + 1 always, and patching alpha0 so that
        // result remains intact.
        // Note that we need in.size.width >= 2, for both sx0 and
        // sx0+1 were indexing pixels inside the input's width.
        if (map.index1 != map.index0 + 1) {
            GAPI_DbgAssert(map.index1 == map.index0);
            GAPI_DbgAssert(in.size.width >= 2);
            if (map.index0 < in.size.width-1) {
                // sx1=sx0+1 fits inside row,
                // make sure alpha0=unity and alpha1=0,
                // so that result equals src[sx0]*unity
                alpha0 = saturate_cast<alpha_type>(unity);
            } else {
                // shift sx0 to left by 1 pixel,
                // and make sure that alpha0=0 and alpha1==1,
                // so that result equals to src[sx0+1]*unity
                alpha0 = 0;
                index0--;
            }
        }

        alpha[x] = alpha0;
        index[x] = index0;

        for (int l = 0; l < 4; l++) {
            clone[4*x + l] = alpha0;
        }
    }

    auto *beta    = scr.beta;
    auto *index_y = scr.mapsy;

    for (int y = 0; y < outSz.height; y++) {
        auto mapY = Mapper::map(vRatio, 0, in.size.height, y);
        beta[y] = mapY.alpha0;
        index_y[y] = mapY.index0;
        index_y[outSz.height + y] = mapY.index1;
    }
}

template<typename T, class Mapper>
static void calcRowLinear(const cv::gapi::fluid::View  & in,
                                cv::gapi::fluid::Buffer& out,
                                cv::gapi::fluid::Buffer& scratch) {
    using alpha_type = typename Mapper::alpha_type;

    auto  inSz =  in.meta().size;
    auto outSz = out.meta().size;

    auto inY = in.y();
    int length = out.length();
    int outY = out.y();
    int lpi = out.lpi();
    GAPI_DbgAssert(outY + lpi <= outSz.height);

    GAPI_DbgAssert(lpi <= 4);

    linearScratchDesc<T, Mapper, 1> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    const auto *alpha = scr.alpha;
    const auto *clone = scr.clone;
    const auto *mapsx = scr.mapsx;
    const auto *beta0 = scr.beta;
    const auto *mapsy = scr.mapsy;
    auto *tmp         = scr.tmp;

    const auto *beta = beta0 + outY;
    const T *src0[4];
    const T *src1[4];
    T *dst[4];

    for (int l = 0; l < lpi; l++) {
        auto index0 = mapsy[outY + l] - inY;
        auto index1 = mapsy[outSz.height + outY + l] - inY;
        src0[l] = in.InLine<const T>(index0);
        src1[l] = in.InLine<const T>(index1);
        dst[l] = out.OutLine<T>(l);
    }

    #ifdef HAVE_AVX512
    if (with_cpu_x86_avx512_core()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 64 && outSz.width >= 32) {
                avx512::calcRowLinear_8UC1(reinterpret_cast<uint8_t**>(dst),
                                           reinterpret_cast<const uint8_t**>(src0),
                                           reinterpret_cast<const uint8_t**>(src1),
                                           reinterpret_cast<const short*>(alpha),
                                           reinterpret_cast<const short*>(clone),
                                           reinterpret_cast<const short*>(mapsx),
                                           reinterpret_cast<const short*>(beta),
                                           reinterpret_cast<uint8_t*>(tmp),
                                           inSz, outSz, lpi);

                return;
            }
        }

        if (std::is_same<T, float>::value) {
            avx512::calcRowLinear_32F(reinterpret_cast<float**>(dst),
                                      reinterpret_cast<const float**>(src0),
                                      reinterpret_cast<const float**>(src1),
                                      reinterpret_cast<const float*>(alpha),
                                      reinterpret_cast<const int*>(mapsx),
                                      reinterpret_cast<const float*>(beta),
                                      inSz, outSz, lpi);
            return;
        }
    }
    #else
    (void)tmp;
    (void)clone;
    #endif

    #ifdef HAVE_AVX2
    if (with_cpu_x86_avx2()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 32 && outSz.width >= 16) {
                avx::calcRowLinear_8UC1(reinterpret_cast<uint8_t**>(dst),
                                        reinterpret_cast<const uint8_t**>(src0),
                                        reinterpret_cast<const uint8_t**>(src1),
                                        reinterpret_cast<const short*>(alpha),
                                        reinterpret_cast<const short*>(clone),
                                        reinterpret_cast<const short*>(mapsx),
                                        reinterpret_cast<const short*>(beta),
                                        reinterpret_cast<uint8_t*>(tmp),
                                        inSz, outSz, lpi);

                return;
            }
        }

        if (std::is_same<T, float>::value) {
            avx::calcRowLinear_32F(reinterpret_cast<float**>(dst),
                                   reinterpret_cast<const float**>(src0),
                                   reinterpret_cast<const float**>(src1),
                                   reinterpret_cast<const float*>(alpha),
                                   reinterpret_cast<const int*>(mapsx),
                                   reinterpret_cast<const float*>(beta),
                                   inSz, outSz, lpi);
            return;
        }
    }
    #endif

    #ifdef HAVE_SSE
    if (with_cpu_x86_sse42()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 16 && outSz.width >= 8) {
                calcRowLinear_8UC1(reinterpret_cast<uint8_t**>(dst),
                                   reinterpret_cast<const uint8_t**>(src0),
                                   reinterpret_cast<const uint8_t**>(src1),
                                   reinterpret_cast<const short*>(alpha),
                                   reinterpret_cast<const short*>(clone),
                                   reinterpret_cast<const short*>(mapsx),
                                   reinterpret_cast<const short*>(beta),
                                   reinterpret_cast<uint8_t*>(tmp),
                                   inSz, outSz, lpi);
                return;
            }
        }

        if (std::is_same<T, float>::value) {
            calcRowLinear_32F(reinterpret_cast<float**>(dst),
                              reinterpret_cast<const float**>(src0),
                              reinterpret_cast<const float**>(src1),
                              reinterpret_cast<const float*>(alpha),
                              reinterpret_cast<const int*>(mapsx),
                              reinterpret_cast<const float*>(beta),
                              inSz, outSz, lpi);
            return;
        }
    }
    #endif  // HAVE_SSE

#ifdef HAVE_NEON
    if (std::is_same<T, uint8_t>::value) {
        if (inSz.width >= 16 && outSz.width >= 8) {
            neon::calcRowLinear_8UC1(reinterpret_cast<uint8_t**>(dst),
                                     reinterpret_cast<const uint8_t**>(src0),
                                     reinterpret_cast<const uint8_t**>(src1),
                                     reinterpret_cast<const short*>(alpha),
                                     reinterpret_cast<const short*>(clone),
                                     reinterpret_cast<const short*>(mapsx),
                                     reinterpret_cast<const short*>(beta),
                                     reinterpret_cast<uint8_t*>(tmp),
                                     inSz, outSz, lpi);
            return;
        }
    }

    if (std::is_same<T, float>::value) {
        neon::calcRowLinear_32F(reinterpret_cast<float**>(dst),
                                reinterpret_cast<const float**>(src0),
                                reinterpret_cast<const float**>(src1),
                                reinterpret_cast<const float*>(alpha),
                                reinterpret_cast<const int*>(mapsx),
                                reinterpret_cast<const float*>(beta),
                                inSz, outSz, lpi);
        return;
    }
#endif

    for (int l = 0; l < lpi; l++) {
        constexpr static const auto unity = Mapper::unity;

        auto beta0 =                                   beta[l];
        auto beta1 = saturate_cast<alpha_type>(unity - beta[l]);

        for (int x = 0; x < length; x++) {
            auto alpha0 =                                   alpha[x];
            auto alpha1 = saturate_cast<alpha_type>(unity - alpha[x]);
            auto sx0 = mapsx[x];
            auto sx1 = sx0 + 1;
            T tmp0 = calc(beta0, src0[l][sx0], beta1, src1[l][sx0]);
            T tmp1 = calc(beta0, src0[l][sx1], beta1, src1[l][sx1]);
            dst[l][x] = calc(alpha0, tmp0, alpha1, tmp1);
        }
    }
}

template<typename T, class Mapper, int numChan>
static void calcRowLinearC(const cv::gapi::fluid::View  & in,
                           std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, numChan>& out,
                           cv::gapi::fluid::Buffer& scratch) {
    using alpha_type = typename Mapper::alpha_type;

    auto  inSz =  in.meta().size;
    auto outSz = out[0].get().meta().size;

    auto inY  = in.y();
    auto outY = out[0].get().y();
    auto lpi  = out[0].get().lpi();

    GAPI_DbgAssert(outY + lpi <= outSz.height);
    GAPI_DbgAssert(lpi <= 4);

    linearScratchDesc<T, Mapper, numChan> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    const auto *alpha = scr.alpha;
    const auto *clone = scr.clone;
    const auto *mapsx = scr.mapsx;
    const auto *beta0 = scr.beta;
    const auto *mapsy = scr.mapsy;
    auto *tmp         = scr.tmp;

    const auto *beta = beta0 + outY;
    const T *src0[4];
    const T *src1[4];
    std::array<std::array<T*, 4>, numChan> dst;

    for (int l = 0; l < lpi; l++) {
        auto index0 = mapsy[outY + l] - inY;
        auto index1 = mapsy[outSz.height + outY + l] - inY;
        src0[l] = in.InLine<const T>(index0);
        src1[l] = in.InLine<const T>(index1);
        for (int c=0; c < numChan; c++) {
            dst[c][l] = out[c].get().template OutLine<T>(l);
        }
    }

#ifdef HAVE_AVX512
    if (with_cpu_x86_avx512_core()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 64 && outSz.width >= 32) {
                avx512::calcRowLinear_8UC<numChan>(dst,
                                                   reinterpret_cast<const uint8_t**>(src0),
                                                   reinterpret_cast<const uint8_t**>(src1),
                                                   reinterpret_cast<const short*>(alpha),
                                                   reinterpret_cast<const short*>(clone),
                                                   reinterpret_cast<const short*>(mapsx),
                                                   reinterpret_cast<const short*>(beta),
                                                   reinterpret_cast<uint8_t*>(tmp),
                                                   inSz, outSz, lpi);
                return;
            }
        }
    }
#else
    (void)tmp;
    (void)clone;
#endif

#ifdef HAVE_AVX2
    if (with_cpu_x86_avx2()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 32 && outSz.width >= 16) {
                avx::calcRowLinear_8UC<numChan>(dst,
                                                reinterpret_cast<const uint8_t**>(src0),
                                                reinterpret_cast<const uint8_t**>(src1),
                                                reinterpret_cast<const short*>(alpha),
                                                reinterpret_cast<const short*>(clone),
                                                reinterpret_cast<const short*>(mapsx),
                                                reinterpret_cast<const short*>(beta),
                                                reinterpret_cast<uint8_t*>(tmp),
                                                inSz, outSz, lpi);
                return;
            }
        }
    }
#endif

#ifdef HAVE_SSE
    if (with_cpu_x86_sse42()) {
        if (std::is_same<T, uint8_t>::value) {
            if (inSz.width >= 16 && outSz.width >= 8) {
                calcRowLinear_8UC<numChan>(dst,
                                           reinterpret_cast<const uint8_t**>(src0),
                                           reinterpret_cast<const uint8_t**>(src1),
                                           reinterpret_cast<const short*>(alpha),
                                           reinterpret_cast<const short*>(clone),
                                           reinterpret_cast<const short*>(mapsx),
                                           reinterpret_cast<const short*>(beta),
                                           reinterpret_cast<uint8_t*>(tmp),
                                           inSz, outSz, lpi);
                return;
            }
        }
    }
#endif  // HAVE_SSE

#ifdef HAVE_NEON
    if (std::is_same<T, uint8_t>::value) {
        if (inSz.width >= 16 && outSz.width >= 8) {
            neon::calcRowLinear_8UC<numChan>(dst,
                                             reinterpret_cast<const uint8_t**>(src0),
                                             reinterpret_cast<const uint8_t**>(src1),
                                             reinterpret_cast<const short*>(alpha),
                                             reinterpret_cast<const short*>(clone),
                                             reinterpret_cast<const short*>(mapsx),
                                             reinterpret_cast<const short*>(beta),
                                             reinterpret_cast<uint8_t*>(tmp),
                                             inSz, outSz, lpi);
            return;
         }
    }
#endif  // HAVE_NEON

    auto length = out[0].get().length();

    for (int l = 0; l < lpi; l++) {
        constexpr static const auto unity = Mapper::unity;

        auto beta0 =                                   beta[l];
        auto beta1 = saturate_cast<alpha_type>(unity - beta[l]);

        for (int x = 0; x < length; x++) {
            auto alpha0 =                                   alpha[x];
            auto alpha1 = saturate_cast<alpha_type>(unity - alpha[x]);
            auto sx0 = mapsx[x];
            auto sx1 = sx0 + 1;

            for (int c = 0; c < numChan; c++) {
                auto idx0 = numChan*sx0 + c;
                auto idx1 = numChan*sx1 + c;
                T tmp0 = calc(beta0, src0[l][idx0], beta1, src1[l][idx0]);
                T tmp1 = calc(beta0, src0[l][idx1], beta1, src1[l][idx1]);
                dst[c][l][x] = calc(alpha0, tmp0, alpha1, tmp1);
            }
        }
    }
}


//------------------------------------------------------------------------------

namespace linear {
struct Mapper {
    typedef short alpha_type;
    typedef short index_type;
    constexpr static const int unity = ONE;

    typedef MapperUnit<short, short> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        float f = static_cast<float>((outCoord + 0.5) * ratio - 0.5);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>(ONE * (1.0f - f));
        u.alpha1 = saturate_cast<short>(ONE *         f);

        return u;
    }
};
}  // namespace linear

namespace linear32f {
struct Mapper {
    typedef float alpha_type;
    typedef int   index_type;
    constexpr static const float unity = 1;

    typedef MapperUnit<float, int> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        float f = static_cast<float>((outCoord + 0.5) * ratio - 0.5);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = 1.f - f;
        u.alpha1 =       f;

        return u;
    }
};
}  // namespace linear32f

namespace areaUpscale {
struct Mapper {
    typedef short alpha_type;
    typedef short index_type;
    constexpr static const int unity = ONE;

    typedef MapperUnit<short, short> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        int s = cvFloor(outCoord*ratio);
        float f = static_cast<float>((outCoord+1) - (s+1)/ratio);
        f = f <= 0 ? 0.f : f - cvFloor(f);

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>(ONE * (1.0f - f));
        u.alpha1 = saturate_cast<short>(ONE *         f);

        return u;
    }
};
}  // namespace areaUpscale

namespace areaUpscale32f {
struct Mapper {
    typedef float alpha_type;
    typedef int   index_type;
    constexpr static const float unity = 1;

    typedef MapperUnit<float, int> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        int s = cvFloor(outCoord*ratio);
        float f = static_cast<float>((outCoord+1) - (s+1)/ratio);
        f = f <= 0 ? 0.f : f - cvFloor(f);

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = 1.0f - f;
        u.alpha1 =        f;

        return u;
    }
};
}  // namespace areaUpscale32f

//------------------------------------------------------------------------------

template<typename A, typename I, typename W>
struct AreaDownMapper {
    typedef A alpha_type;
    typedef I index_type;
    typedef W  work_type;

    typedef MapperUnit<alpha_type, index_type> Unit;

    inline Unit map(int outCoord) {
        double inCoord0 =  outCoord      * ratio;
        double inCoord1 = (outCoord + 1) * ratio;

        double index0 = std::floor(inCoord0 + 0.001);
        double index1 =  std::ceil(inCoord1 - 0.001);

        double alpha0 =   (index0 + 1 - inCoord0) * inv_ratio;
        double alpha1 = - (index1 - 1 - inCoord1) * inv_ratio;

        GAPI_Assert(0 <= outCoord && outCoord <= outSz-1);
        GAPI_Assert(0 <= index0 && index0 < index1 && index1 <= inSz);

        Unit unit;

        unit.index0 = checked_cast<index_type>(index0);
        unit.index1 = checked_cast<index_type>(index1);

        unit.alpha0 = convert_cast<alpha_type>(alpha0);
        unit.alpha1 = convert_cast<alpha_type>(alpha1);

        return unit;
    }

    int    inSz, outSz;
    double ratio, inv_ratio;

    alpha_type  alpha;  // == inv_ratio, rounded

    void init(int _inSz, int _outSz) {
        inSz  = _inSz;
        outSz = _outSz;

        inv_ratio = invRatio(inSz, outSz);
        ratio     = 1.0 / inv_ratio;

        alpha = convert_cast<alpha_type>(inv_ratio);
    }
};

namespace areaDownscale32f {
struct Mapper: public AreaDownMapper<float, int, float> {
    Mapper(int _inSz, int _outSz) {
        init(_inSz, _outSz);
    }
};
}

namespace areaDownscale8u {
struct Mapper: public AreaDownMapper<Q0_16, short, Q8_8> {
    Mapper(int _inSz, int _outSz) {
        init(_inSz, _outSz);
    }
};
}

template<typename Mapper>
static void initScratchArea(const cv::GMatDesc& in, const Size& outSz,
                            cv::gapi::fluid::Buffer &scratch) {
    using Unit = typename Mapper::Unit;
    using alpha_type = typename Mapper::alpha_type;
    using index_type = typename Mapper::index_type;

    // compute the chunk of input pixels for each output pixel,
    // along with the coefficients for taking the weigthed sum

    Size inSz = in.size;
    Mapper mapper(inSz.width, outSz.width);

    std::vector<Unit> xmaps(outSz.width);
    int  maxdif = 0;

    for (int w = 0; w < outSz.width; w++) {
        Unit map = mapper.map(w);
        xmaps[w] = map;

        int dif = map.index1 - map.index0;
        if (dif > maxdif)
            maxdif = dif;
    }

    // This assertion is critical for our trick with chunk sizes:
    // we would expand a chunk it is is smaller than maximal size
    GAPI_Assert(inSz.width >= maxdif);

    // pack the input chunks positions and coefficients into scratch-buffer,
    // along with the maximal size of chunk (note that chunk size may vary)

    size_t scratch_bytes =               sizeof(int)
                         + outSz.width * sizeof(index_type)
                         + outSz.width * sizeof(alpha_type) * maxdif
                         +  inSz.width * sizeof(alpha_type);
    Size scratch_size{static_cast<int>(scratch_bytes), 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    auto *maxdf =  scratch.OutLine<int>();
    auto *index = reinterpret_cast<index_type*>(maxdf + 1);
    auto *alpha = reinterpret_cast<alpha_type*>(index + outSz.width);
//  auto *vbuf  = reinterpret_cast<work_type *>(alpha + outSz.width * maxdif);

    for (int w = 0; w < outSz.width; w++) {
        // adjust input indices so that:
        // - data chunk is exactly maxdif pixels
        // - data chunk fits inside input width
        int index0 = xmaps[w].index0;
        int index1 = xmaps[w].index1;
        int i0 = index0, i1 = index1;
        i1 = (std::min)(i0 + maxdif, in.size.width);
        i0 =            i1 - maxdif;
        GAPI_DbgAssert(i0 >= 0);

        // fulfill coefficients for the data chunk,
        // extending with zeros if any extra pixels
        alpha_type *alphaw = &alpha[w * maxdif];
        for (int i = 0; i < maxdif; i++) {
            if (i + i0 == index0) {
                alphaw[i] = xmaps[w].alpha0;

            } else if (i + i0 == index1 - 1) {
                alphaw[i] = xmaps[w].alpha1;

            } else if (i + i0 > index0 && i + i0 < index1 - 1) {
                alphaw[i] = mapper.alpha;

            } else {
                alphaw[i] = 0;
            }
        }

        // start input chunk with adjusted position
        index[w] = i0;
    }

    *maxdf = maxdif;
}

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

template<typename T, typename Mapper>
static void calcAreaRow(const cv::gapi::fluid::View& in, cv::gapi::fluid::Buffer& out,
                              cv::gapi::fluid::Buffer& scratch) {
    using Unit = typename Mapper::Unit;
    using alpha_type = typename Mapper::alpha_type;
    using index_type = typename Mapper::index_type;
    using  work_type = typename Mapper::work_type;

    Size inSz  =  in.meta().size;
    Size outSz = out.meta().size;

    // this method is valid only for down-scale
    GAPI_DbgAssert(inSz.width  >= outSz.width);
    GAPI_DbgAssert(inSz.height >= outSz.height);

//  Mapper xmapper(inSz.width,  outSz.width);
    Mapper ymapper(inSz.height, outSz.height);

    auto *xmaxdf = scratch.OutLine<const int>();
    auto  maxdif = xmaxdf[0];

    auto *xindex = reinterpret_cast<const index_type*>(xmaxdf + 1);
    auto *xalpha = reinterpret_cast<const alpha_type*>(xindex + outSz.width);
    auto *vbuf_c = reinterpret_cast<const  work_type*>(xalpha + outSz.width * maxdif);

    auto *vbuf = const_cast<work_type*>(vbuf_c);

    int iny = in.y();
    int y = out.y();

    int lpi = out.lpi();
    GAPI_DbgAssert(y + lpi <= outSz.height);

    for (int l = 0; l < lpi; l++) {
        Unit ymap = ymapper.map(y + l);

        GAPI_Assert(ymap.index1 - ymap.index0 <= 32);
        GAPI_Assert(ymap.index1 - ymap.index0 > 0);
        const T *src[32] = {};

        for (int yin = ymap.index0; yin < ymap.index1; yin++) {
            src[yin - ymap.index0] = in.InLine<const T>(yin - iny);
        }

        auto dst = out.OutLine<T>(l);

        #ifdef HAVE_AVX512
        if (with_cpu_x86_avx512f()) {
            if (std::is_same<T, uchar>::value) {
                avx512::calcRowArea_8U(reinterpret_cast<uchar*>(dst),
                                       reinterpret_cast<const uchar**>(src),
                                       inSz, outSz,
                                       static_cast<Q0_16>(ymapper.alpha),
                                       reinterpret_cast<const MapperUnit8U&>(ymap),
                                       xmaxdf[0],
                                       reinterpret_cast<const short*>(xindex),
                                       reinterpret_cast<const Q0_16*>(xalpha),
                                       reinterpret_cast<Q8_8*>(vbuf));
                continue;  // next l = 0, ..., lpi-1
            }

            if (std::is_same<T, float>::value) {
                avx512::calcRowArea_32F(reinterpret_cast<float*>(dst),
                                        reinterpret_cast<const float**>(src),
                                        inSz, outSz,
                                        static_cast<float>(ymapper.alpha),
                                        reinterpret_cast<const MapperUnit32F&>(ymap),
                                        xmaxdf[0],
                                        reinterpret_cast<const int*>(xindex),
                                        reinterpret_cast<const float*>(xalpha),
                                        reinterpret_cast<float*>(vbuf));
                continue;
            }
        }
        #endif  // HAVE_AVX512

        #ifdef HAVE_AVX2
        if (with_cpu_x86_avx2()) {
            if (std::is_same<T, uchar>::value) {
                avx::calcRowArea_8U(reinterpret_cast<uchar*>(dst),
                                    reinterpret_cast<const uchar**>(src),
                                    inSz, outSz,
                                    static_cast<Q0_16>(ymapper.alpha),
                                    reinterpret_cast<const MapperUnit8U&>(ymap),
                                    xmaxdf[0],
                                    reinterpret_cast<const short*>(xindex),
                                    reinterpret_cast<const Q0_16*>(xalpha),
                                    reinterpret_cast<Q8_8*>(vbuf));
                continue;  // next l = 0, ..., lpi-1
            }

            if (std::is_same<T, float>::value) {
                avx::calcRowArea_32F(reinterpret_cast<float*>(dst),
                                     reinterpret_cast<const float**>(src),
                                     inSz, outSz,
                                     static_cast<float>(ymapper.alpha),
                                     reinterpret_cast<const MapperUnit32F&>(ymap),
                                     xmaxdf[0],
                                     reinterpret_cast<const int*>(xindex),
                                     reinterpret_cast<const float*>(xalpha),
                                     reinterpret_cast<float*>(vbuf));
                continue;
            }
        }
        #endif  // HAVE_AVX2

        #ifdef HAVE_SSE
        if (with_cpu_x86_sse42()) {
            if (std::is_same<T, uchar>::value) {
                calcRowArea_8U(reinterpret_cast<uchar*>(dst),
                               reinterpret_cast<const uchar**>(src),
                               inSz, outSz,
                               static_cast<Q0_16>(ymapper.alpha),
                               reinterpret_cast<const MapperUnit8U&>(ymap),
                               xmaxdf[0],
                               reinterpret_cast<const short*>(xindex),
                               reinterpret_cast<const Q0_16*>(xalpha),
                               reinterpret_cast<Q8_8*>(vbuf));
                continue;  // next l = 0, ..., lpi-1
            }

            if (std::is_same<T, float>::value) {
                calcRowArea_32F(reinterpret_cast<float*>(dst),
                                reinterpret_cast<const float**>(src),
                                inSz, outSz,
                                static_cast<float>(ymapper.alpha),
                                reinterpret_cast<const MapperUnit32F&>(ymap),
                                xmaxdf[0],
                                reinterpret_cast<const int*>(xindex),
                                reinterpret_cast<const float*>(xalpha),
                                reinterpret_cast<float*>(vbuf));
                continue;
            }
        }
        #endif  // HAVE_SSE

        // vertical pass
        int y_1st = ymap.index0;
        int ylast = ymap.index1 - 1;
        if (y_1st < ylast) {
            for (int w = 0; w < inSz.width; w++) {
                vbuf[w] = mulas(ymap.alpha0, src[0][w])        // Q8_8 = Q0_16 * U8
                        + mulas(ymap.alpha1, src[ylast - y_1st][w]);
            }

            for (int i = 1; i < ylast - y_1st; i++) {
                for (int w = 0; w < inSz.width; w++) {
                    vbuf[w] += mulas(ymapper.alpha, src[i][w]);
                }
            }
        } else {
            for (int w = 0; w < inSz.width; w++) {
                vbuf[w] = convert_cast<work_type>(src[0][w]);  // Q8_8 = U8
            }
        }

        // horizontal pass
        for (int x = 0; x < outSz.width; x++) {
            work_type sum = 0;

            auto        index =  xindex[x];
            const auto *alpha = &xalpha[x * maxdif];

            for (int i = 0; i < maxdif; i++) {
                sum +=  mulaw(alpha[i], vbuf[index + i]);      // Q8_8 = Q0_16 * Q8_8
            }

            dst[x] = convert_cast<T>(sum);                     // U8 = Q8_8
        }
    }
}

#if defined __GNUC__
# pragma GCC diagnostic pop
#endif

//----------------------------------------------------------------------

#if USE_CVKL

// taken from: ie_preprocess_data.cpp
static int getResizeAreaTabSize(int dst_go, int ssize, int dsize, float scale) {
    static const float threshold = 1e-3f;
    int max_count = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        if (sx1 - fsx1 > threshold) {
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            count++;
        }

        if (fsx2 - sx2 > threshold) {
            count++;
        }
        max_count = (std::max)(max_count, count);
    }

    return max_count;
}

// taken from: ie_preprocess_data.cpp
static void computeResizeAreaTab(int src_go, int dst_go, int ssize, int dsize, float scale,
                                 uint16_t* si, uint16_t* alpha, int max_count) {
    static const float threshold = 1e-3f;
    int k = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;
        float cellWidth = (std::min)(scale, ssize - fsx1);

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        si[col - dst_go] = (uint16_t)(sx1 - src_go);

        if (sx1 - fsx1 > threshold) {
            si[col - dst_go] = (uint16_t)(sx1 - src_go - 1);
            alpha[k++] = (uint16_t)((1 << 16) * ((sx1 - fsx1) / cellWidth));
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            alpha[k++] = (uint16_t)((1 << 16) * (1.0f / cellWidth));
            count++;
        }

        if (fsx2 - sx2 > threshold) {
            alpha[k++] = (uint16_t)((1 << 16) * ((std::min)((std::min)(fsx2 - sx2, 1.f), cellWidth) / cellWidth));
            count++;
        }

        if (count != max_count) {
            alpha[k++] = 0;
        }
    }
}

// teken from: ie_preprocess_data.cpp
static void generate_alpha_and_id_arrays(int x_max_count, int dcols, const uint16_t* xalpha, uint16_t* xsi,
                                         uint16_t** alpha, uint16_t** sxid) {
    if (x_max_count <= 4) {
        for (int col = 0; col < dcols; col++) {
            for (int x = 0; x < x_max_count; x++) {
                alpha[x][col] = xalpha[col*x_max_count + x];
            }
        }
    }
    if (x_max_count <= 4) {
        for (int col = 0; col <= dcols - 8; col += 8) {
            for (int chunk_num_h = 0; chunk_num_h < x_max_count; chunk_num_h++) {
                for (int i = 0; i < 128 / 16; i++) {
                    int id_diff = xsi[col + i] - xsi[col];

                    for (int chunk_num_v = 0; chunk_num_v < x_max_count; chunk_num_v++) {
                        uint16_t* sxidp = sxid[chunk_num_v] + col * x_max_count + chunk_num_h * 8;

                        int id0 = (id_diff + chunk_num_v) * 2 + 0;
                        int id1 = (id_diff + chunk_num_v) * 2 + 1;

                        (reinterpret_cast<int8_t*>(sxidp + i))[0] = static_cast<int8_t>(id0 >= (chunk_num_h * 16) && id0 < (chunk_num_h + 1) * 16 ? id0 : -1);
                        (reinterpret_cast<int8_t*>(sxidp + i))[1] = static_cast<int8_t>(id1 >= (chunk_num_h * 16) && id1 < (chunk_num_h + 1) * 16 ? id1 : -1);
                    }
                }
            }
        }
    }
}

// taken from: ie_preprocess_data.cpp
// (and simplified for specifically downscale area 8u)
static size_t resize_get_buffer_size(const Size& inSz, const Size& outSz) {
    int dst_full_width  = outSz.width;
    int dst_full_height = outSz.height;
    int src_full_width  =  inSz.width;
    int src_full_height =  inSz.height;

    auto resize_area_u8_downscale_sse_buffer_size = [&]() {
        const int dwidth  = outSz.width;
        const int dheight = outSz.height;
        const int swidth  =  inSz.width;

        const int dst_go_x = 0;
        const int dst_go_y = 0;

        int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width,  dwidth,  static_cast<float>(src_full_width)  / dst_full_width)  + 1;
        int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, static_cast<float>(src_full_height) / dst_full_height) + 1;

        size_t si_buf_size = sizeof(uint16_t) * dwidth + sizeof(uint16_t) * dheight;
        size_t alpha_buf_size =
                sizeof(uint16_t) * (dwidth * x_max_count + 8 * 16) + sizeof(uint16_t) * dheight * y_max_count;
        size_t vert_sum_buf_size = sizeof(uint16_t) * (swidth * 2);
        size_t alpha_array_buf_size = sizeof(uint16_t) * 4 * dwidth;
        size_t sxid_array_buf_size = sizeof(uint16_t) * 4 * 4 * dwidth;

        size_t buffer_size = si_buf_size +
                             alpha_buf_size +
                             vert_sum_buf_size +
                             alpha_array_buf_size +
                             sxid_array_buf_size;

        return buffer_size;
    };

    return resize_area_u8_downscale_sse_buffer_size();
}

// buffer-fulfill is taken from: ie_preprocess_data_sse42.cpp
static void initScratchArea_CVKL_U8(const cv::GMatDesc & in,
                                    const       Size   & outSz,
                               cv::gapi::fluid::Buffer & scratch) {
    const Size& inSz = in.size;

    // estimate buffer size
    size_t scratch_bytes = resize_get_buffer_size(inSz, outSz);

    // allocate buffer

    Size scratch_size{static_cast<int>(scratch_bytes), 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    // fulfil buffer
    {
        // this code is taken from: ie_preprocess_data_sse42.cpp
        // (and simplified for 1-channel cv::Mat instead of blob)

        auto dwidth  = outSz.width;
        auto dheight = outSz.height;
        auto swidth  =  inSz.width;
        auto sheight =  inSz.height;

        const int src_go_x = 0;
        const int src_go_y = 0;
        const int dst_go_x = 0;
        const int dst_go_y = 0;

        auto src_full_width  = swidth;
        auto src_full_height = sheight;
        auto dst_full_width  = dwidth;
        auto dst_full_height = dheight;

        float scale_x = static_cast<float>(src_full_width)  / dst_full_width;
        float scale_y = static_cast<float>(src_full_height) / dst_full_height;

        int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width,  dwidth,  scale_x);
        int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, scale_y);

        auto* maxdif = scratch.OutLine<int>();
        auto* xsi = reinterpret_cast<uint16_t*>(maxdif + 2);
        auto* ysi = xsi + dwidth;
        auto* xalpha = ysi + dheight;
        auto* yalpha = xalpha + dwidth*x_max_count + 8*16;
    //  auto* vert_sum = yalpha + dheight*y_max_count;

        maxdif[0] = x_max_count;
        maxdif[1] = y_max_count;

        computeResizeAreaTab(src_go_x, dst_go_x, src_full_width,   dwidth, scale_x, xsi, xalpha, x_max_count);
        computeResizeAreaTab(src_go_y, dst_go_y, src_full_height, dheight, scale_y, ysi, yalpha, y_max_count);

        int vest_sum_size = 2*swidth;
        uint16_t* vert_sum = yalpha + dheight*y_max_count;
        uint16_t* alpha0 = vert_sum + vest_sum_size;
        uint16_t* alpha1 = alpha0 + dwidth;
        uint16_t* alpha2 = alpha1 + dwidth;
        uint16_t* alpha3 = alpha2 + dwidth;
        uint16_t* sxid0 = alpha3 + dwidth;
        uint16_t* sxid1 = sxid0 + 4*dwidth;
        uint16_t* sxid2 = sxid1 + 4*dwidth;
        uint16_t* sxid3 = sxid2 + 4*dwidth;

        uint16_t* alpha[] = {alpha0, alpha1, alpha2, alpha3};
        uint16_t* sxid[] = {sxid0, sxid1, sxid2, sxid3};
        generate_alpha_and_id_arrays(x_max_count, dwidth, xalpha, xsi, alpha, sxid);
    }
}

static void calcAreaRow_CVKL_U8(const cv::gapi::fluid::View   & in,
                                      cv::gapi::fluid::Buffer & out,
                                      cv::gapi::fluid::Buffer & scratch) {
    Size inSz  =  in.meta().size;
    Size outSz = out.meta().size;

    // this method is valid only for down-scale
    GAPI_DbgAssert(inSz.width  >= outSz.width);
    GAPI_DbgAssert(inSz.height >= outSz.height);

    int dwidth  = outSz.width;
    int dheight = outSz.height;

    auto* maxdif = scratch.OutLine<int>();
    int x_max_count = maxdif[0];
    int y_max_count = maxdif[1];

    auto* xsi = reinterpret_cast<uint16_t*>(maxdif + 2);
    auto* ysi    = xsi + dwidth;
    auto* xalpha = ysi + dheight;
    auto* yalpha = xalpha + dwidth*x_max_count + 8*16;
    auto* vert_sum = yalpha + dheight*y_max_count;

    int iny =  in.y();
    int   y = out.y();

    int lpi = out.lpi();
    GAPI_DbgAssert(y + lpi <= outSz.height);

    for (int l = 0; l < lpi; l++) {
        int yin0 = ysi[y + l];
        int yin1 = yin0 + y_max_count;

        GAPI_Assert(yin1 - yin0 <= 32);
        const uint8_t *src[32] = {};

        for (int yin = yin0; yin < yin1 && yin < inSz.height; yin++) {
            if (yalpha[(y+l)*y_max_count + yin - yin0] == 0) {
                src[yin - yin0] = in.InLine<const uint8_t>(yin - iny - 1);
            } else {
                src[yin - yin0] = in.InLine<const uint8_t>(yin - iny);
            }
        }

        uint8_t *dst = out.OutLine<uint8_t>(l);

        calcRowArea_CVKL_U8_SSE42(src, dst, inSz, outSz, y + l, xsi, ysi,
                      xalpha, yalpha, x_max_count, y_max_count, vert_sum);
    }
}

#endif  // CVKL
//----------------------------------------------------------------------

GAPI_FLUID_KERNEL(FScalePlane8u, ScalePlane8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, linear::Mapper>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
        calcRowLinear<uint8_t, linear::Mapper>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlanes, ScalePlanes, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in, int, Size,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, linear::Mapper, 3>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, int, Size, Size/*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out1,
                    cv::gapi::fluid::Buffer& out2,
                    cv::gapi::fluid::Buffer& out3,
                    cv::gapi::fluid::Buffer& scratch) {
        constexpr int numChan = 3;
        std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, numChan> out = {out1, out2, out3};
        calcRowLinearC<uint8_t, linear::Mapper, numChan>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlanes4, ScalePlanes4, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in, int, Size,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, linear::Mapper, 4>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, int, Size, Size/*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out1,
                    cv::gapi::fluid::Buffer& out2,
                    cv::gapi::fluid::Buffer& out3,
                    cv::gapi::fluid::Buffer& out4,
                    cv::gapi::fluid::Buffer& scratch) {
        constexpr int numChan = 4;
        std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, numChan> out = {out1, out2, out3, out4};
        calcRowLinearC<uint8_t, linear::Mapper, numChan>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FUpscalePlaneArea8u, UpscalePlaneArea8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, areaUpscale::Mapper>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
        calcRowLinear<uint8_t, areaUpscale::Mapper>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FUpscalePlaneArea32f, UpscalePlaneArea32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<float, areaUpscale32f::Mapper>(in, outSz, scratch, 0);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
        calcRowLinear<float, areaUpscale32f::Mapper>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlane32f, ScalePlane32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);

        initScratchLinear<float, linear32f::Mapper>(in, outSz, scratch, 0);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
        calcRowLinear<float, linear32f::Mapper>(in, out, scratch);
    }
};

//----------------------------------------------------------------------

GAPI_FLUID_KERNEL(FScalePlaneArea32f, ScalePlaneArea32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchArea<areaDownscale32f::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
        calcAreaRow<float, areaDownscale32f::Mapper>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlaneArea8u, ScalePlaneArea8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
    #if USE_CVKL
        #ifdef HAVE_SSE
        if (with_cpu_x86_sse42()) {
            const Size& inSz = in.size;
            if (inSz.width > outSz.width && inSz.height > outSz.height) {
                // CVKL code we use supports only downscale
                initScratchArea_CVKL_U8(in, outSz, scratch);
                return;
            }
        }
        #endif  // HAVE_SSE
    #endif

        initScratchArea<areaDownscale8u::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer &scratch) {
    #if USE_CVKL
        #ifdef HAVE_SSE
        if (with_cpu_x86_sse42()) {
            auto  inSz =  in.meta().size;
            auto outSz = out.meta().size;
            if (inSz.width > outSz.width && inSz.height > outSz.height) {
                // CVKL's code supports only downscale
                calcAreaRow_CVKL_U8(in, out, scratch);
                return;
            }
        }
        #endif  // HAVE_SSE
    #endif

        calcAreaRow<uint8_t, areaDownscale8u::Mapper>(in, out, scratch);
    }
};

static const int ITUR_BT_601_CY = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;
static const int ITUR_BT_601_SHIFT = 20;

static inline void uvToRGBuv(const uchar u, const uchar v, int& ruv, int& guv, int& buv) {
    int uu, vv;
    uu = static_cast<int>(u) - 128;
    vv = static_cast<int>(v) - 128;

    ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * vv;
    guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu;
    buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * uu;
}

static inline void yRGBuvToRGB(const uchar vy, const int ruv, const int guv, const int buv,
                                uchar& r, uchar& g, uchar& b) {
    int yy = static_cast<int>(vy);
    int y = std::max(0, yy - 16) * ITUR_BT_601_CY;
    r = saturate_cast<uchar>((y + ruv) >> ITUR_BT_601_SHIFT);
    g = saturate_cast<uchar>((y + guv) >> ITUR_BT_601_SHIFT);
    b = saturate_cast<uchar>((y + buv) >> ITUR_BT_601_SHIFT);
}

static void calculate_nv12_to_rgb_fallback(const  uchar **y_rows,
                                           const  uchar *uv_row,
                                                  uchar **out_rows,
                                           int buf_width) {
    for (int i = 0; i < buf_width; i += 2) {
        uchar u = uv_row[i];
        uchar v = uv_row[i + 1];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uchar vy = y_rows[y][i + x];
                uchar r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                out_rows[y][3*(i + x)]     = r;
                out_rows[y][3*(i + x) + 1] = g;
                out_rows[y][3*(i + x) + 2] = b;
            }
        }
    }
}

static void calculate_i420_to_rgb_fallback(const  uchar **y_rows,
                                           const  uchar *u_row,
                                           const  uchar *v_row,
                                                  uchar **out_rows,
                                           int buf_width) {
    for (int i = 0; i < buf_width; i += 2) {
        uchar u = u_row[i / 2];
        uchar v = v_row[i / 2];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uchar vy = y_rows[y][i + x];
                uchar r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                out_rows[y][3*(i + x)]     = r;
                out_rows[y][3*(i + x) + 1] = g;
                out_rows[y][3*(i + x) + 2] = b;
            }
        }
    }
}

GAPI_FLUID_KERNEL(FNV12toRGB, NV12toRGB, false) {
    static const int Window = 1;
    static const int LPI    = 2;
    static const auto Kind = cv::GFluidKernel::Kind::YUV420toRGB;

    static void run(const cv::gapi::fluid::View &in_y,
                    const cv::gapi::fluid::View &in_uv,
                          cv::gapi::fluid::Buffer &out) {
        const uchar* uv_row = in_uv.InLineB(0);
        const uchar* y_rows[2] = {in_y. InLineB(0), in_y. InLineB(1)};
        uchar* out_rows[2] = {out.OutLineB(0), out.OutLineB(1)};

        int buf_width = out.length();

// AVX512 implementation of wide universal intrinsics is slower than AVX2.
// It is turned off until the cause isn't found out.
    #if 0
    #ifdef HAVE_AVX512
        if (with_cpu_x86_avx512_core()) {
            #define CV_AVX_512DQ 1
            avx512::calculate_nv12_to_rgb(y_rows, uv_row, out_rows, buf_width);
            return;
        }
    #endif  // HAVE_AVX512
    #endif

    #ifdef HAVE_AVX2
        if (with_cpu_x86_avx2()) {
            avx::calculate_nv12_to_rgb(y_rows, uv_row, out_rows, buf_width);
            return;
        }
    #endif  // HAVE_AVX2
    #ifdef HAVE_SSE
        if (with_cpu_x86_sse42()) {
            calculate_nv12_to_rgb(y_rows, uv_row, out_rows, buf_width);
            return;
        }
    #endif  // HAVE_SSE

    #ifdef HAVE_NEON
        neon::calculate_nv12_to_rgb(y_rows, uv_row, out_rows, buf_width);
        return;
    #endif  // HAVE_NEON

        calculate_nv12_to_rgb_fallback(y_rows, uv_row, out_rows, buf_width);
    }
};

GAPI_FLUID_KERNEL(FI420toRGB, I420toRGB, false) {
    static const int Window = 1;
    static const int LPI    = 2;
    static const auto Kind = cv::GFluidKernel::Kind::YUV420toRGB;

    static void run(const cv::gapi::fluid::View &in_y,
                    const cv::gapi::fluid::View &in_u,
                    const cv::gapi::fluid::View &in_v,
                          cv::gapi::fluid::Buffer &out) {
        const uchar* u_row = in_u.InLineB(0);
        const uchar* v_row = in_v.InLineB(0);
        const uchar* y_rows[2] = {in_y. InLineB(0), in_y. InLineB(1)};
        uchar* out_rows[2] = {out.OutLineB(0), out.OutLineB(1)};

        int buf_width = out.length();
        GAPI_DbgAssert(in_u.length() ==  in_v.length());

        // AVX512 implementation of wide universal intrinsics is slower than AVX2.
        // It is turned off until the cause isn't found out.
        #if 0
        #ifdef HAVE_AVX512
            if (with_cpu_x86_avx512_core()) {
               #define CV_AVX_512DQ 1
               avx512::calculate_i420_to_rgb(y_rows, u_row, v_row, out_rows, buf_width);
               return;
            }
        #endif  // HAVE_AVX512
        #endif

        #ifdef HAVE_AVX2
            if (with_cpu_x86_avx2()) {
               avx::calculate_i420_to_rgb(y_rows, u_row, v_row, out_rows, buf_width);
               return;
            }
        #endif  // HAVE_AVX2
        #ifdef HAVE_SSE
            if (with_cpu_x86_sse42()) {
               calculate_i420_to_rgb(y_rows, u_row, v_row, out_rows, buf_width);
               return;
            }
        #endif  // HAVE_SSE

        #ifdef HAVE_NEON
            neon::calculate_i420_to_rgb(y_rows, u_row, v_row, out_rows, buf_width);
            return;
        #endif  // HAVE_NEON

        calculate_i420_to_rgb_fallback(y_rows, u_row, v_row, out_rows, buf_width);
    }
};

namespace {

template <typename src_t, typename dst_t>
void convert_precision(const uint8_t* src, uint8_t* dst, const int width) {
    const auto *in  = reinterpret_cast<const src_t *>(src);
            auto *out = reinterpret_cast<dst_t *>(dst);

    for (int i = 0; i < width; i++) {
        out[i] = saturate_cast<dst_t>(in[i]);
    }
}

}  // namespace

GAPI_FLUID_KERNEL(FConvertDepth, ConvertDepth, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View& src, int depth, cv::gapi::fluid::Buffer& dst) {
        GAPI_Assert(src.meta().depth == CV_8U || src.meta().depth == CV_32F || src.meta().depth == CV_16U);
        GAPI_Assert(dst.meta().depth == CV_8U || dst.meta().depth == CV_32F || dst.meta().depth == CV_16U);
        GAPI_Assert(src.meta().chan == 1);
        GAPI_Assert(dst.meta().chan == 1);
        GAPI_Assert(src.length() == dst.length());

        constexpr unsigned supported_types_n = 3;
        using p_f = void (*)( const uint8_t* src,  uint8_t* dst, const int width);
        using table_string_t = std::array<p_f, supported_types_n>;

        constexpr std::array<table_string_t, supported_types_n> func_table = {
                table_string_t{convert_precision<uint16_t, uint16_t>, convert_precision<uint16_t, float>, convert_precision<uint16_t, uint8_t>},
                table_string_t{convert_precision<float,    uint16_t>, convert_precision<float,    float>, convert_precision<float,    uint8_t>},
                table_string_t{convert_precision<uint8_t,  uint16_t>, convert_precision<uint8_t,  float>, convert_precision<uint8_t,  uint8_t>}
        };

        auto depth_to_index = [](int depth){
            switch (depth) {
                case  CV_16U: return 0;
                case  CV_32F: return 1;
                case  CV_8U:  return 2;
                default: GAPI_Assert(!"not supported depth"); return -1;
            }
        };
        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();
        auto const src_index = depth_to_index(src.meta().depth);
        auto const dst_index = depth_to_index(dst.meta().depth);

        (func_table[src_index][dst_index])(in, out, width);
    }
};

namespace {
    template <typename src_t, typename dst_t>
    void sub(const uint8_t* src, uint8_t* dst, const int width, double c) {
        const auto *in  = reinterpret_cast<const src_t *>(src);
              auto *out = reinterpret_cast<dst_t *>(dst);

        for (int i = 0; i < width; i++) {
            out[i] = saturate_cast<dst_t>(in[i] - c);
        }
    }

    template <typename src_t, typename dst_t>
    void div(const uint8_t* src, uint8_t* dst, const int width, double c) {
        const auto *in  = reinterpret_cast<const src_t *>(src);
              auto *out = reinterpret_cast<dst_t *>(dst);

        for (int i = 0; i < width; i++) {
            out[i] = saturate_cast<dst_t>(in[i] / c);
        }
    }
}  // namespace

GAPI_FLUID_KERNEL(FSubC, GSubC, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View& src, const cv::Scalar &scalar, int depth, cv::gapi::fluid::Buffer& dst) {
        GAPI_Assert(src.meta().depth == CV_32F && src.meta().chan == 1);

        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();

        sub<float, float>(in, out, width, scalar[0]);
    }
};

GAPI_FLUID_KERNEL(FDivC, GDivC, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View &src, const cv::Scalar &scalar, double _scale, int /*dtype*/,
            cv::gapi::fluid::Buffer &dst) {
        GAPI_Assert(src.meta().depth == CV_32F && src.meta().chan == 1);

        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();

        div<float, float>(in, out, width, scalar[0]);
    }
};
}  // namespace kernels

//----------------------------------------------------------------------

using namespace kernels;

cv::gapi::GKernelPackage preprocKernels() {
    return cv::gapi::kernels
        < FChanToPlane
        , FScalePlanes
        , FScalePlanes4
        , FScalePlane
        , FScalePlane32f
        , FScalePlane8u
        , FUpscalePlaneArea8u
        , FUpscalePlaneArea32f
        , FScalePlaneArea8u
        , FScalePlaneArea32f
        , FMerge2
        , FMerge3
        , FMerge4
        , FSplit2
        , FSplit3
        , FSplit4
        , FNV12toRGB
        , FI420toRGB
        , FConvertDepth
        , FSubC
        , FDivC
        >();
}

}  // namespace gapi
}  // namespace InferenceEngine
