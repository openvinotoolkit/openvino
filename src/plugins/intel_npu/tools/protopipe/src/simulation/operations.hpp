//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/cpu/gcpukernel.hpp>  // GAPI_OCV_KERNEL
#include <opencv2/gapi/gkernel.hpp>         // G_API_OP
#include <opencv2/gapi/infer.hpp>

#include "utils/data_providers.hpp"
#include "utils/utils.hpp"

// clang-format off
struct InferCall {
    cv::GProtoArgs operator()(const cv::GProtoArgs& inputs);

    std::string              tag;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

struct DummyState { };
struct GDummyM {
    static const char *id() { return "custom.dummym"; }
    static std::vector<cv::GMat> on(const std::vector<cv::GMat>           &ins,
                                    const uint64_t                        delay_in_us,
                                    const std::vector<IDataProvider::Ptr> &providers,
                                    const bool                            disable_copy);
    static cv::GMetaArgs getOutMeta(const cv::GMetaArgs&, const cv::GArgs &args);
};

struct GCPUDummyM: public cv::detail::KernelTag {
    using API = GDummyM;
    using State = DummyState;

    static cv::gapi::GBackend backend();
    static cv::GCPUKernel kernel();
    static void setup(const cv::GMetaArgs    &metas,
                      cv::GArgs              gargs,
                      cv::GArg               &state,
                      const cv::GCompileArgs &args);
    static void call(cv::GCPUContext &ctx);
};

struct DummyCall {
    std::vector<IDataProvider::Ptr> providers;
    uint64_t delay_in_us;
    // NB: Don't pull data from providers if enabled
    bool disable_copy = false;
    cv::GProtoArgs operator()(const cv::GProtoArgs& inputs);
};

using F = std::function<void()>;

G_TYPED_KERNEL(GCompound, <cv::GMat(cv::GMat, F)>, "custom.compound")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, F){
        return in;
    }
};

GAPI_OCV_KERNEL(GCPUCompound, GCompound)
{
    static void run(const cv::Mat& in,
                    F function,
                    cv::Mat& out)
    {
        function();
    }
};

struct CompoundCall {
    cv::GProtoArgs operator()(const cv::GProtoArgs& inputs);
    F function;
};
