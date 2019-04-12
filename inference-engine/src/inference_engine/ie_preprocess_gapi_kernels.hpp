// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

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
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, int) {
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
        static std::tuple<cv::GMatDesc, cv::GMatDesc, cv::GMatDesc> outMeta(const cv::GMatDesc &in, int /*type*/, const Size &szIn, const Size &szOut, int) {
            cv::GMatDesc out_desc;
            out_desc.depth = in.depth;
            out_desc.chan  = 1;
            out_desc.size = szOut;
            return std::make_tuple(out_desc, out_desc, out_desc);
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

    cv::gapi::GKernelPackage preprocKernels();

}  // namespace gapi
}  // namespace InferenceEngine
