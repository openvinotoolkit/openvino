//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/gcompiled.hpp>   // cv::GCompiled
#include <opencv2/gapi/gstreaming.hpp>  // cv::GStreamingCompiled

#include "scenario/criterion.hpp"

class PipelinedExecutor {
public:
    explicit PipelinedExecutor(cv::GStreamingCompiled&& compiled);

    struct Output {
        uint64_t elapsed_us;
    };
    using Callback = std::function<bool(cv::GStreamingCompiled&)>;

    Output runLoop(cv::GRunArgs&& inputs, Callback callback, ITermCriterion::Ptr criterion);

private:
    cv::GStreamingCompiled m_compiled;
};

class SyncExecutor {
public:
    explicit SyncExecutor(cv::GCompiled&& compiled);

    struct Output {
        uint64_t elapsed_us;
    };
    using Callback = std::function<bool(cv::GCompiled&)>;

    Output runLoop(Callback callback, ITermCriterion::Ptr criterion);
    void reset();

private:
    cv::GCompiled m_compiled;
};
