// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

# ifndef GAPI_STANDALONE
# error non standalone GAPI
# endif

#include <tuple>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace InferenceEngine {
namespace gapi {
    using Size = cv::gapi::own::Size;

    using GMat2 = std::tuple<cv::GMat, cv::GMat>;
    using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
    using GMat4 = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat>;

    G_TYPED_KERNEL(ChanToPlane, <cv::GMat(cv::GMat, int)>, "com.intel.ie.chan_to_plane") {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, int chan) {
            GAPI_Assert(chan < in.chan);
            return in.withType(in.depth, 1);
        }
    };

    G_TYPED_KERNEL(ScalePlane, <cv::GMat(cv::GMat, int, Size, Size, int)>, "com.intel.ie.scale_plane") {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, int type, const Size &szIn, const Size &szOut, int) {
            GAPI_Assert(type == in.depth);
            return in.withSize(szOut);
        }
    };

    G_TYPED_KERNEL_M(ScalePlanes, <GMat3(cv::GMat, int, Size, Size, int)>, "com.intel.ie.scale_planes") {
        static std::tuple<cv::GMatDesc, cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc &in, int /*type*/, const Size &szIn,
                                                                            const Size &szOut, int interp) {
            // This kernel supports only RGB 8U inputs
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 3);
            // cv::INTER_LINEAR is the only supported interpolation
            GAPI_Assert(interp == cv::INTER_LINEAR);
            // ad-hoc withChan
            cv::GMatDesc out_desc = in.withType(in.depth, 1).withSize(szOut);
            return std::make_tuple(out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL_M(ScalePlanes4, <GMat4(cv::GMat, int, Size, Size, int)>, "com.intel.ie.scale_planes4") {
        static std::tuple<cv::GMatDesc, cv::GMatDesc, cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc &in, int /*type*/, const Size &szIn,
                                                                            const Size &szOut, int interp) {
            // This kernel supports only RGB 8U inputs
            GAPI_Assert(in.depth == CV_8U);
            GAPI_Assert(in.chan == 4);
            // cv::INTER_LINEAR is the only supported interpolation
            GAPI_Assert(interp == cv::INTER_LINEAR);
            // ad-hoc withChan
            cv::GMatDesc out_desc = in.withType(in.depth, 1).withSize(szOut);
            return std::make_tuple(out_desc, out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL(ScalePlane8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_8u") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(ScalePlane32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_32f") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(UpscalePlaneArea8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.upscale_plane_area_8u") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
            GAPI_DbgAssert(in.size.width < sz.width || in.size.height < sz.height);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(UpscalePlaneArea32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.upscale_plane_area_32f") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
            GAPI_DbgAssert(in.size.width < sz.width || in.size.height < sz.height);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(ScalePlaneArea8u, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_area_8u") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_8U && in.chan == 1);
            GAPI_DbgAssert(in.size.width >= sz.width && in.size.height >= sz.height);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(ScalePlaneArea32f, <cv::GMat(cv::GMat, Size, int)>, "com.intel.ie.scale_plane_area_32f") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const Size & sz, int) {
            GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);
            GAPI_DbgAssert(in.size.width >= sz.width && in.size.height >= sz.height);
            return in.withSize(sz);
        }
    };

    G_TYPED_KERNEL(Merge2, <cv::GMat(cv::GMat, cv::GMat)>, "com.intel.ie.merge2") {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) {
            // FIXME: check a/b are equal!
            return in.withType(in.depth, 2);
        }
    };

    G_TYPED_KERNEL(Merge3, <cv::GMat(cv::GMat, cv::GMat, cv::GMat)>, "com.intel.ie.merge3") {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &, const cv::GMatDesc &) {
            // FIXME: check a/b are equal!
            return in.withType(in.depth, 3);
        }
    };

    G_TYPED_KERNEL(Merge4, <cv::GMat(cv::GMat, cv::GMat, cv::GMat, cv::GMat)>, "com.intel.ie.merge4") {
        static cv::GMatDesc outMeta(const cv::GMatDesc& in,
                                    const cv::GMatDesc&, const cv::GMatDesc&, const cv::GMatDesc&) {
            // FIXME: check a/b are equal!
            return in.withType(in.depth, 4);
        }
    };

    G_TYPED_KERNEL_M(Split2, <GMat2(cv::GMat)>, "com.intel.ie.split2") {
        static std::tuple<cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc& in) {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL_M(Split3, <GMat3(cv::GMat)>, "com.intel.ie.split3") {
        static std::tuple<cv::GMatDesc, cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc& in) {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL_M(Split4, <GMat4(cv::GMat)>, "com.intel.ie.split4") {
        static std::tuple<cv::GMatDesc, cv::GMatDesc, cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc& in) {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL(NV12toRGB, <cv::GMat(cv::GMat, cv::GMat)>, "com.intel.ie.nv12torgb") {
        static cv::GMatDesc outMeta(cv::GMatDesc in_y, cv::GMatDesc in_uv) {
            GAPI_Assert(in_y.chan == 1);
            GAPI_Assert(in_uv.chan == 2);
            GAPI_Assert(in_y.depth == CV_8U);
            GAPI_Assert(in_uv.depth == CV_8U);
            // UV size should be aligned with Y
            GAPI_Assert(in_y.size.width == 2 * in_uv.size.width);
            GAPI_Assert(in_y.size.height == 2 * in_uv.size.height);
            return in_y.withType(CV_8U, 3);
        }
    };

    G_TYPED_KERNEL(I420toRGB, <cv::GMat(cv::GMat, cv::GMat, cv::GMat)>, "com.intel.ie.i420torgb") {
        static cv::GMatDesc outMeta(cv::GMatDesc in_y, cv::GMatDesc in_u, cv::GMatDesc in_v) {
            GAPI_Assert(in_y.chan == 1);
            GAPI_Assert(in_u.chan == 1);
            GAPI_Assert(in_v.chan == 1);
            GAPI_Assert(in_y.depth == CV_8U);
            GAPI_Assert(in_u.depth == CV_8U);
            GAPI_Assert(in_v.depth == CV_8U);
            // U and V size should be aligned with Y
            GAPI_Assert(in_y.size.width  == 2 * in_u.size.width);
            GAPI_Assert(in_y.size.height == 2 * in_u.size.height);

            GAPI_Assert(in_y.size.width  == 2 * in_v.size.width);
            GAPI_Assert(in_y.size.height == 2 * in_v.size.height);

            return in_y.withType(CV_8U, 3);
        }
    };

    G_TYPED_KERNEL(ConvertDepth, <cv::GMat(cv::GMat, int depth)>, "com.intel.ie.ConvertDepth") {
        static cv::GMatDesc outMeta(const cv::GMatDesc& in, int depth) {
            GAPI_Assert(in.depth == CV_8U || in.depth == CV_16U || in.depth == CV_32F);
            GAPI_Assert(depth == CV_8U || depth == CV_32F || depth == CV_16U);

            return in.withDepth(depth);
        }
    };

    G_TYPED_KERNEL(GSubC, <cv::GMat(cv::GMat, cv::GScalar, int)>, "com.intel.ie.math.subC") {
        static cv::GMatDesc outMeta(cv::GMatDesc a, cv::GScalarDesc, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GDivC, <cv::GMat(cv::GMat, cv::GScalar, double, int)>, "com.intel.ie.math.divC") {
        static cv::GMatDesc outMeta(cv::GMatDesc a, cv::GScalarDesc, double, int ddepth) {
            return a.withDepth(ddepth);
        }
    };
    cv::gapi::GKernelPackage preprocKernels();


namespace kernels {

struct fp_16_t {
    int16_t v;
};

template<typename type>
struct cv_type_to_depth;

template<> struct cv_type_to_depth<std::uint8_t> { enum { depth = CV_8U }; };
template<> struct cv_type_to_depth<std::int8_t> { enum { depth = CV_8S }; };
template<> struct cv_type_to_depth<std::uint16_t> { enum { depth = CV_16U }; };
template<> struct cv_type_to_depth<std::int16_t> { enum { depth = CV_16S }; };
template<> struct cv_type_to_depth<std::int32_t> { enum { depth = CV_32S }; };
template<> struct cv_type_to_depth<float> { enum { depth = CV_32F }; };
template<> struct cv_type_to_depth<fp_16_t> { enum { depth = CV_16F }; };

template<typename ... types>
struct typelist {};

template<typename type_list>
struct head;

template<template<typename ...> class list, typename head_t, typename ... types>
struct head<list<head_t, types...>> { using type = head_t; };

template<typename typelist>
using head_t = typename head<typelist>::type;

template<typename type>
struct type_to_type {};

template <typename typelist>
struct type_dispatch_impl;

//FIXME: add test for type_dispatch
template <template<typename ...> class typelist, typename... type>
struct type_dispatch_impl<typelist<type...>> {
    template <typename result_t, typename default_t, typename type_id_t, typename type_to_id_t, typename type_to_value_t>
    static result_t dispatch(type_id_t type_id, type_to_id_t&& type_to_id, type_to_value_t&& type_to_value, default_t default_value) {
        result_t res = default_value;

        bool matched = false;
        std::initializer_list<int>({
            !matched && (type_id == type_to_id(type_to_type<type>{})) ?
                    (matched = true, res = type_to_value(type_to_type<type>{})), 0
                    : 0
            ...
            });
        return res;
    }

    template <typename result_t, typename default_t, typename pred_t, typename type_to_value_t>
    static result_t dispatch(pred_t&& pred, type_to_value_t&& type_to_value, default_t default_value) {
        result_t res = default_value;

        bool matched = false;
        std::initializer_list<int>({
            !matched && pred(type_to_type<type>{}) ?
                    (matched = true, res = type_to_value(type_to_type<type>{})), 0
                    : 0
            ...
            });
        return res;
    }
};

template<typename left_typelsist, typename right_typelsist>
struct concat;

template<typename left_typelsist, typename right_typelsist>
using concat_t = typename concat<left_typelsist, right_typelsist>::type;

template<template<typename ...> class left_list, typename ... left_types, template<typename ...> class right_list, typename ... right_types>
struct concat<left_list<left_types...>, right_list<right_types...>> {
    using type = left_list<left_types..., right_types...>;
};

template< class T, class U >
using is_same_t = typename std::is_same<T, U>::type;

template<bool C, class T, class E> struct if_c_impl;

template<class T, class E> struct if_c_impl<true, T, E> {
    using type = T;
};

template<class T, class E> struct if_c_impl<false, T, E> {
    using type = E;
};

template<bool C, class T, class E>
using if_c = typename if_c_impl<C, T, E>::type;

template<class C, class T, class E>
using if_ = typename if_c_impl<C::value != 0, T, E>::type;

template<typename typelist, typename type>
struct remove;

template<typename typelist, typename type>
using remove_t = typename remove<typelist, type>::type;


template<template<typename ...> class list, typename head_t, typename ... types, typename t>
struct remove<list<head_t, types...>, t> {
    using type = concat_t<
        if_<is_same_t<head_t, t>, list<>, list<head_t>>,
        remove_t<list<types...>, t>
    >;
};

template<template<typename ...> class list, typename t>
struct remove<list<>, t> {
    using type = list<>;
};

template <typename typelist, typename default_t, typename type_id_t, typename type_to_id_t, typename type_to_value_t,
    typename result_t = decltype(std::declval<type_to_value_t>()(type_to_type<head_t<typelist>> {})) >
    inline result_t type_dispatch(type_id_t type_id, type_to_id_t&& type_to_id, type_to_value_t&& type_to_value, default_t default_value = {}) {
    return type_dispatch_impl<typelist>::template dispatch<result_t>(std::forward<type_id_t>(type_id),
        std::forward<type_to_id_t>(type_to_id),
        std::forward<type_to_value_t>(type_to_value),
        std::forward<default_t>(default_value));
}

template <typename typelist, typename default_t, typename pred_t, typename type_to_value_t,
    typename result_t = decltype(std::declval<type_to_value_t>()(type_to_type<head_t<typelist>> {})) >
    inline result_t type_dispatch(pred_t&& pred, type_to_value_t&& type_to_value, default_t default_value = {}) {
    return type_dispatch_impl<typelist>::template dispatch<result_t>(std::forward<pred_t>(pred),
        std::forward<type_to_value_t>(type_to_value),
        std::forward<default_t>(default_value));
}

namespace {
struct cv_type_id {
    template <typename type>
    const int operator()(type_to_type<type>) { return cv_type_to_depth<type>::depth; }
};

}  // namespace

template <typename typelist>
bool is_cv_type_in_list(const int type_id) {
    return type_dispatch<typelist>(type_id, cv_type_id{}, [](...) { return true; }, false);
}
}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
