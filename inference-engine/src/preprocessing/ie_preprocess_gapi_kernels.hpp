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

}  // namespace gapi
}  // namespace InferenceEngine
